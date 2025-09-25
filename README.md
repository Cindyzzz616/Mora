# Mora: authentic language acquisition from user-generated short-form videos
<div style="display: flex; align-items: center;">
    <img src="you-jujube/src/assets/yuo.svg" alt="YouLingo Logo" style="height: 120px;" />
    <img src="you-jujube/src/assets/youlingo.svg" alt="YouLingo Logo" style="height: 100px;" />
</div>

Many language learners struggle to find materials that align with both their  interests and proficiency level. YouLingo addresses this gap by leveraging the popularity and appeal of short-form videos, with premises of authentic materials, comprehensible input, and an emphasis on listening and speaking. It allows users to generate content in their native language and interact with target language videos made by others.Throughout stages of design, development, testing, and eventual commercialization, YouLingo translates evidence-based principles into a self-sustaining language exchange platform.

## Contributors
### UofTHacks team: the YouLinGOATs
- Adrien Mery
  - https://github.com/NOBODIDI
- Rui Weng
  - https://github.com/r-weng
- Lucie Yang
  - https://github.com/lucieyang1
- Cindy Zhang
  - https://github.com/Cindyzzz616

### MVP building: Roi Digital
- Nirdesh Shrestha
  - https://github.com/Nightking25
- Shishir Shrestha
  - https://github.com/sthz31

## Table of Contents
1. [About the Project](#1-about-the-project)
2. [Theoretical Background](#2-theoretical-background)
3. [Features](#3-features)
4. [Technologies Used](#4-technologies-used)
5. [Getting Started](#5-getting-started)
6. [Potential Improvements](#6-potential-improvements)

## 1. About the Project

This project was initially created for the UofTHacks 12 Hackathon and was the winner of Best University of Toronto Hack.

The team's take on this year's theme, Perspective, is an application that takes a different look
at media consumption beyond a form of entertainment, and challenges pre-conceived ideas about language learning - that it has to be confined to
textbooks and classrooms. The result of this is YouLingo, a web application designed to help users learn new languages by making YouTube videos into 
custom language lessons. Users can explore various videos, add their own video URLs, and manage their favorite videos.

### From Hackathon Idea to Research-Driven Innovation

Since then, YouLingo has grown into a research-backed language learning project grounded in both theory and practice. In the summer of 2025, its conceptual foundation was developed through two parallel research programs that explored how to evaluate and modify the difficulty of videos for language learners—specifically focusing on what affects intelligibility, comprehensibility, and vocabulary retention.

### Dual Research Streams

One research stream was funded by the University of Toronto Excellence Award ($7500) and supervised by Professor Lyn Tieu in the Department of French. This project focused on lexical-syntactic variables, including lexical coverage, vocabulary size, and pause structure.

The second was part of the Research Opportunity Program at UTM, conducted under Professor Jessamyn Schertz in the Sounds of UTM Lab. Here, the focus shifted to acoustic-phonological variables such as speech rate, phonemic overlap, and phonological neighborhood density. Together, these studies built a rigorous theoretical model of video difficulty that directly informs YouLingo’s next phase.

### A Shift Toward User-Generated Short-Form Content

Drawing from insights about learner engagement and current digital trends, YouLingo has pivoted toward a novel and, to our knowledge, unprecedented design in the language learning space: leveraging user-generated short-form videos, similar to those seen on platforms like TikTok or Instagram Reels. While many apps use pre-recorded content or scripted dialogues, no existing language learning app currently centers its experience around short-form videos created by users themselves. This approach harnesses the widespread popularity, accessibility, and entertainment value of short-form content, while embedding it directly into a pedagogical framework.

In this model, users would record short videos in their native language—sharing personal stories, cultural insights, or everyday moments—and watch videos in their target language created by speakers of that language. This creates a mutual language exchange environment, where learners are not only passive recipients of curated input but active participants in a multilingual community. The design also encourages interaction through comment threads, enabling users to ask questions, clarify meaning, and build connections across linguistic boundaries. Altogether, the platform merges authentic input, peer-to-peer learning, and community-driven content—a combination that promises both pedagogical depth and high user engagement.

### Current Development with Roi Digital

YouLingo is now moving from theory to practice. An MVP (Minimum Viable Product) is currently in development in partnership with [Roi Digital](https://roidigital.ca/), a Quebec-based software development company. The MVP will incorporate research insights on difficulty modeling and serve as a testing ground for new features and potential follow-up studies.

### Public Showcase

YouLingo’s summer research and MVP will be publicly showcased at the [Experiential Education Fair at the University of Toronto Mississauga](https://www.utm.utoronto.ca/experience/events/experiential-education-fair) on September 25, 2025, marking its first major milestone on the path from hackathon prototype to real-world application.

### Looking Ahead: New Research and Engagement Tools

Next steps include a follow-up study to further validate the difficulty model within the context of the app and to begin tackling YouLingo’s second core research question:
“How do we facilitate language acquisition based on our materials?”
This will involve the design of interactive practice and engagement features tailored to the cognitive and motivational needs of learners.

_The sections below pertain to the version of the application developed during UofTHacks._

## 2. Theoretical Background
### Hypotheses of Language Acquisition
This application is based on the language acquisition theory posited by Stephen Krashen, a renowned American linguist who has written more than 500 articles and books on the second language acquisition and related fields. [1]
In his theory, he outlined six hypotheses about how to effectively gain fluency in a second language. They are:  
1. **The Acquisition-Learning Hypothesis:** learning a language involves consciously obtaining knowledge _about_ a language, while the
process of acquisition is unconscious and involves direct interaction _with_ a language. The former approach is used in formal
language education, which often yields unsatisfactory results. In comparison, the latter approach is the process through which
children come to master their native language.
2. **The Monitor Hypothesis:** when we are consciously learning a language, there is an internal monitor that judges and corrects the every
utterance. As a result, fluency is sacrificed to maximize accuracy, which hinders the progress of the learner in the long run.
3. **The Natural Order hypothesis:** children first acquire the comprehension capacities, then production capacities; they also master languages
in spoken forms before learning the written forms. Therefore, the natural order of language acquisition is _listening_, _speaking_, _reading_,
then _writing_. There are also orders of acquiring different grammatical structures. When adults follow the same order, they are able to
communicate more fluently and effectively.
4. **The Input Hypothesis:** language input can only be helpful to a learner when they are comprehensible. When they wish to progress,
they must take small steps by consuming content that encompass the previous material with small additions. This can be described by the
analogy "i+1", where i represents the current knowledge and 1 represents the small addition on that basis.
5. **The Affective Filter Hypothesis:** when learners are in stressful surroundings, their brains filter out the input
and render it useless. Their interest in the content they consume can also affect how well they internalize the language. 
Therefore, it is important for them to be in a relaxed environment where they can engage with the materials of their choosing and make mistakes.
6. **The Reading Hypothesis:** our vocabulary expands when we read materials from real-life contexts, rather than engage in rote
memorization of vocabulary lists. [2]
These hypotheses have proven to be effective from their significant transformation of language instruction and successful results.

### Our Implementation
YouLingo instantiates these hypotheses in concrete and innovative ways:
1. It does not explicitly instruct users to learn and memorize grammar rules, conjugations and vocabulary lists. Instead, it provides
users with natural input sourced from the wealth of videos on YouTube, which are created by real individuals in real-life contexts.
This is similar to how a child picks up their native language from listening to conversations in their surroundings.
2. When a user is focused on the content of their favourite videos, they are subconsciously taking in their target language without
pressure to perform in a grammatically impeccable manner.
3. Our application focuses on the first stage of the language acquisition process, since that is the key stage that is often skipped
or skimmed through. When users are ready, it also provides the potential for them to practice other aspects of the language - for
example, they can read the transcripts and hone their production abilities with the quizzes the app generates. Overall, they
are grounded in the practice of content consumption to ensure sufficient input.
4. This hypothesis is implemented by the flagship feature of our application, the recommendation system that compares videos with the user's
knowledge base using a variety of quantifiable linguistic metrics.
5. Similar to the second hypothesis, this principle emphasizes the importance of a good environment and emotional state in efficacy
of acquisition. Our application provides content tailored to the user's preferences and expertise, so they can feel at ease with
the subject of the materials.
6. While our application does not focus on reading, the principles from the sixth hypothesis can be generalized to listening materials.
As mentioned, we provide users with a large amount of video data that strategically introduce new vocabulary in real-life contexts.

### Why CEFR?
At the present stage, we assess the language level of users and the difficulty of videos using the CEFR framework. This system is chosen
because it is used around the world and is a shorthand for quantifying the linguistic complexity of materials. It has six broad levels
as outlined below:
- **A1:** Can understand and use familiar everyday expressions and very basic phrases aimed at the satisfaction of needs of a concrete type. Can introduce him/herself and others and can ask and answer questions about personal details such as where he/she lives, people he/she knows and things he/she has. Can interact in a simple way provided the other person talks slowly and clearly and is prepared to help.
- **A2:** Can understand sentences and frequently used expressions related to areas of most immediate relevance (e.g. very basic personal and family information, shopping, local geography, employment). Can communicate in simple and routine tasks requiring a simple and direct exchange of information on familiar and routine matters.  Can describe in simple terms aspects of his/her background, immediate environment and matters in areas of immediate need.
- **B1:** Can understand the main points of clear standard input on familiar matters regularly encountered in work, school, leisure, etc. Can deal with most situations likely to arise whilst travelling in an area where the language is spoken.  Can produce simple connected text on topics which are familiar or of personal interest. Can describe experiences and events, dreams, hopes & ambitions and briefly give reasons and explanations for opinions and plans.
- **B2:** Can understand the main ideas of complex text on both concrete and abstract topics, including technical discussions in his/her field of specialisation. Can interact with a degree of fluency and spontaneity that makes regular interaction with native speakers quite possible without strain for either party. Can produce clear, detailed text on a wide range of subjects and explain a viewpoint on a topical issue giving the advantages and disadvantages of various options.
- **C1:** Can understand a wide range of demanding, longer texts, and recognise implicit meaning. Can express him/herself fluently and spontaneously without much obvious searching for expressions. Can use language flexibly and effectively for social, academic and professional purposes. Can produce clear, well-structured, detailed text on complex subjects, showing controlled use of organisational patterns, connectors and cohesive devices.
- **C2:** Can understand with ease virtually everything heard or read. Can summarise information from different spoken and written sources, reconstructing arguments and accounts in a coherent presentation. Can express him/herself spontaneously, very fluently and precisely, differentiating finer shades of meaning even in more complex situations. [3]  

In the application, these levels are represented by floating point numbers, which makes it possible to more precisely specify the difficulty of a video, word, tense or clause to a greater precision. A1 is quantified as 1.0, A2 as 2.0 and so on, up to C2 as 5.0.

### Sources  
[1] https://rossier.usc.edu/faculty-research/directory/stephen-krashen  
[2] https://sites.ualberta.ca/~obilash/krashen.html#:~:text=Application%20for%20Teaching-,The%20Acquisition%2DLearning%20hypothesis,of%20just%20'learning'%20it  
[3] https://www.coe.int/en/web/common-european-framework-reference-languages/level-descriptions

## 3. Features
_NOTE: due to time constraints, some of these features have not been fully implemented. The following is a description of the intended design of the application
with the core features available for demonstration._

### User Database
YouLingo allows users to input topics of interest as tags, and save videos as a list. It also stores a range of linguistic competency data for each user. This includes:
* Overall CEFR level: This is stored as a floating point number from 0.0 to 5.0, representing levels from A1 to C2. Upon sign-up, the user 
* Lexicon: This is a list of the vocabulary that the user has accumulated from videos they have watched. Each word has an associated CEFR score, and the application keeps
a tally of the number of times the user has encountered that word.
* Record of Tenses: Similar to the Lexicon, this is a record of the tenses the user has encountered. The following tenses are tracked by the application:

  | Available Tenses              |                           |
  |-------------------------------|---------------------------|
  | Infinitive                    | Present simple            |
  | Infinitive passive            | Past simple               |
  | Infinitive continuous         | Present continuous        |
  | Infinitive perfect            | Present perfect           |
  | Infinitive perfect passive    | Past perfect              |
  | Infinitive perfect continuous | Present perfect continuous |
  | Unbound modal verb            | Present perfect passive   |
  | Imperative                    | Present continuous passive |
  | Gerund simple                 | Gerund perfect            |
  | Past participle simple        | Past perfect continuous   |
  | Gerund perfect passive        | Past perfect passive      |
  | Gerund perfect continuous     | Past continuous passive   |
  | Gerund perfect passive        |                           |

* Record of Clauses: This is a record of the types of clauses the user has encountered. They include:

  | Available Clause Types        |
  |-------------------------------|
  | Adjective clauses             |
  | Adverbial clauses             |
  | Relative clauses              |
  | Noun clauses                  |
  | Sentences inside parentheses  |

[intended but not implemented] Like the lexicon, the record of the types of tenses and clauses also keeps track of the number of times they have been encountered by the user.
Each time the user watches a video, new words, tenses and clause types are added to their knowledge base, and the tally for existing entries increases.
As the user accumulate experience, their overall CEFR level increases automatically based on the tally and average CEFR score of the words in their lexicon, as well as 
their familiarity with tenses and clause types. This in turn increases the average difficulty of videos recommended to them, as discussed below.

### Video Recommendation
The user's home page is populated by a series of recommended videos based on the user's topics of interest, their CEFR level and [intended but
not implemented] the similarity between the video's language content and the user's knowledge base. The precision of the recommendations can be
adjusted (for example, users can choose to view videos with a general difficulty level that is within ±0.5 CEFR-score of their personal level,
and with a word list that has an 80% overlap with their lexicon of familiar vocabulary). The videos are ranked based on these metrics, and new videos are shown upon refreshing the page.

### Viewing Videos
Each video is accompanied by the following components:
- A transcript in the user's target language, and a translation into the language of the user's choice.
- [intended but not implemented] A matching score that indicates the video's similarity to the 
linguistic information that the user has already encountered. This breaks down into three categories:
  - vocabulary
  - tenses
  - clause types
with the same set of available tenses and clause types as the one in the user database.
- A general difficulty score, rated in terms of CEFR levels. This is also expanded into the same categories as above.
- A counter that keeps track of the number of times the user has watched the video.
- A button to save the video.
  
### Saving Videos
Saved videos can be retrieved under a "My Videos" tab. This encourages users to review videos to ensure sufficient repetition.
[intended but not implemented] Videos would be considered as complete after a custom number of replays and the user would be 
allowed to add newer, more challenging videos to their rotation.

### Quizzing
A short quiz is generated for each video, and the user can input their answer to be assessed on their reading and writing abilities.


## 4. Technologies Used
### Libraries
- React
- React Router
- Reactstrap
- Auth0 React
- Bootstrap
- React Icons

### APIs
- CEFR Statistics API by Cathoven
  - https://enterpriseapi.cathoven.com/cefr
- YouTube Transcript API by jedpoix
  - https://github.com/jdepoix/youtube-transcript-api
- YouTube Data API v.3 by Google
  - https://www.googleapis.com/youtube/v3
- Google Cloud Firestore
  - https://cloud.google.com/firestore
- OpenAI GPT-4 API
  - https://platform.openai.com

## 5. Getting Started
### Prerequisites
Make sure you have the following installed on your machine:

- Node.js
- npm (Node Package Manager)
- Python
- pip (Python Package Installer)

### Installation
1. Clone the repository:

```sh
git clone https://github.com/your-username/YouLingo.git
cd YouLingo/you-jujube
```

2. Install the dependencies for the frontend:

```sh
npm install
```

3. Install the dependencies for the backend:

```sh
cd ../flask_backend
pip install -r requirements.txt
```

### Running the Application
#### Running the Frontend

To run the frontend application in a development environment, use the following command:

```sh
cd ../you-jujube
npm start
```

This will start the development server and open the application in your default web browser. The application will automatically reload if you make any changes to the source code.

#### Running the Backend
To run the Flask backend, use the following command:

```sh
cd ../flask_backend
flask run
```

This will start the Flask server. Make sure you have your `serviceAccountKey.json` file in the `flask_backend` directory and your `.env` file properly configured.

### Folder Structure
- `src/`: Contains the source code for the application.
  - `components/`: Contains reusable React components.
  - `services/`: Contains service files for API calls.
  - `App.js`: The main application component.
  - `Home.js`: The home page component.
  - `Explore.js`: The explore page component.
  - `VideoLesson.js`: The video lesson page component.
  - `MyVideos.js`: The my videos page component.
- `flask_backend/`: Contains the Flask backend code.
  - `app.py`: The main Flask application file.
  - `requirements.txt`: The dependencies for the Flask application.

### Authentication
This application uses Auth0 for authentication. Make sure to configure your Auth0 credentials in the application.

## 6. Potential Improvements
There are some features we did not have time to implement, but we look forward to growing this project in the future!
- Transcript highlighting: the transcript can be colour-coded according to the user's familiarity with the words, as well
as words that are far beyond the user's current level. The same can be done with difficult sentence structures and
grammatical formulations. Based on that, YouLingo will recommend parts of the video that the user should review and 
parts that can be skipped over.
- Review reminder: YouLingo can remind the user to re-watch the videos at scheduled intervals, using principles like
the Ebbinghaus forgetting curve.
- Connection to YouTube: for users who are willing, YouLingo can be linked to their YouTube account, so they get even more 
personalized recommendations based on their watch history and behaviours.
- Like YouTube, YouLingo can display the most-viewed segments when hovering over the progress bar.
- There can be further AI integration for advanced features like stitching together lines from different videos to make
custom materials, or generating audio/video scripts altogether.

### Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request.
