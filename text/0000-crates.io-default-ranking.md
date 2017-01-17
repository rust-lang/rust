- Feature Name: crates_io_default_ranking
- Start Date: 2016-12-19
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Crates.io has many useful libraries for a variety of purposes, but it's
difficult to find which crates are meant for a particular purpose and then to
decide among the available crates which one is most suitable in a particular
context. [Categorization][cat-pr] and [badges][badge-pr] are coming to
crates.io; categories help with finding a set of crates to consider and badges
help communicate attributes of crates.

**This RFC aims to create a default ranking of crates within a list of crates
that have a category or keyword in order to make a recommendation to crate users
about which crates are likely to deserve further manual evaluation.**

[cat-pr]: https://github.com/rust-lang/crates.io/pull/473
[badge-pr]: https://github.com/rust-lang/crates.io/pull/481

# Motivation
[motivation]: #motivation

Finding and evaluating crates can be time consuming. People already familiar
with the Rust ecosystem often know which crates are best for which puproses, but
we want to share that knowledge with everyone. For example, someone looking for
a crate to help create a parser should be able to navigate to a category
for that purpose and get a list of crates to consider. This list would include
crates such as [nom][] and [peresil][], and the order in which they appear
should be significant and should help make the decision between the crates in
this category easier.

[nom]: https://crates.io/crates/nom
[peresil]: https://crates.io/crates/peresil

This helps address the goal of "Rust should provide easy access to high quality
crates" as stated in the [Rust 2017 Roadmap][roadmap].

[roadmap]: https://github.com/rust-lang/rfcs/pull/1774

# Detailed design
[design]: #detailed-design

Please see the [Appendix: Comparative Research][comparative-research] section
for ways that other package manager websites have solved this problem, and the
[Appendix: User Research][user-research] section for results of a user research
survey we did on how people evaluate crates by hand today.

A few assumptions we made:

- Measures that can be made automatically are preferred over measures that
  would need administrators, curators, or the community to spend time on
  manually.
- Measures that can be made for any crate regardless of that crate's choice of
  version control, repository host, or CI service are preferred over measures
  that would only be available or would be more easily available with git,
  GitHub, Travis, and Appveyor. Our thinking is that when this additional
  information is available, it would be better to display a badge indicating it
  since this is valuable information, but it should not influence the ranking
  of the crates.
- There are some measures, like "suitability for the current task" or "whether
  I like the way the crate is implemented" that crates.io shouldn't even
  attempt to assess, since those could potentially differ across situations for
  the same person looking for a crate.
- We assume we will be able to calculate these in a reasonable amount of time
  either on-demand or by a background job initiated on crate publish and saved
  in the database as appropriate. We think the measures we have proposed can be
  done without impacting the performance of either publishing or browsing
  crates noticeably. If this does not turn out to be the case, we will have to
  adjust the formula.

## Factors

Through [the survey we conducted][user-research], we found that when people
evaluate crates, they are looking primarily for approximate signals of:

- Ease of use
- Maintenance
- Quality

Cited as secondary signals that were used to infer that the primary signals are
good as well:

- Popularity
- Credibility

We detail how we propose to address each of these in turn, plus a rating of the
five crates from the user research survey as examples.

### Ease of use

By far, the most common attribute people said they considered in the survey was
whether a crate had good documentation. Frequently mentioned when discussing
documentation was the desire to quickly find an example of how to use the crate.

This would be addressed through human evaluation, rather than automatic
evaluation, in two ways:

1. [Render README files on a crate's page on crates.io][render-readme] so that
   people can quickly see for themselves the information that a crate author
   chooses to make available in their README. We can nudge towards having an
   example in the README by adding a template README that includes an Examples
   section [in what `cargo new` generates][cargo-new].
2. Add a mechanism for logged-in crates.io users to indicate that a crate has
   particularly good documentation.
   - This would be a very constrained form of voting/rating: one UI element
     (ex: an up arrow, a thumbs up, a star, a checkbox, a link) that could be
     toggled from "not indicated" to "this crate has good documentation" and
     vice versa.
   - The number of people who have indicated a crate has good documentation
     would be displayed for each crate.
   - That number would be limited to an amount of time (proposal: 6 mo). 6 mo
     after you voted, your vote would disappear and you could choose to renew
     your vote. This would prevent older crates from getting too much of an
     advantage or a high rating being inaccurate if many new, undocumented
     features get added to a crate.
   - This would not influence ranking at all, and therefore is less likely to
     be gamed. You'd need to make many github accounts to easily game this.
   - Since there is no negative "this crate has bad documenation" indication,
     nor is there free-form text, the moderation burden should be minimal.

[render-readme]: https://github.com/rust-lang/crates.io/issues/81
[cargo-new]: https://github.com/rust-lang/cargo/issues/3506

### Maintenance (and Popularity)

The number of releases in the last 6 months and the number of downloads in the
last 90 days can be combined into an automatic indicator of the status of a
crate. This would be more like a badge and would not influence ranking at all.

- Many recent releases and few downloads indicates an *experimental* crate.
- At least occasional releases in the last 6 months and many recent downloads
  indicates a *mainstream* crate.
- Few to no releases in the last 6 months and few recent downloads indicates an
  *inactive* crate.

In table form:

|                | Many releases | Few releases |
|----------------|---------------|--------------|
| Many downloads | Mainstream    | Mainstream   |
| Few downloads  | Experimental  | Inactive     |

TODO: Decide what the cutoff values these measures should have, which we will do
if people are generally in favor of this idea.

By using the number of downloads, crates that are "finished" and stable should
still be regarded as mainstream while many people continue to use it.

These labels will have an indicator of their meaning and how they are calculated
when you hover over them.

A downside of this method is that it does not convey crate author *intent*,
only what one might assume based only on these two measures. A crate might get
popular while an author still considers it to be experimental, thus creating
expectations of stability and support.

We might need to experiment with the thresholds for the number of releases and
the number of downloads considered to be "few" and "many".

Alternatives:

- Also factor in the version number: keep the "Experimental" label unless a
  crate version has many downloads *and* its version number is >= 1.0.0. This
  might be better once more crates release a 1.0.0 version.
- Don't show any label for the "Mainstream" category and only label
  "Experimental" or "Inactive" crates.
- Use different words for these concepts
- Use more words for these concepts that more clearly states what is measured:
  - "This crate has many recent releases and many downloads"
  - "This crate has many recent downloads but has not been updated in the last 6
    months"
  - "This crate has many recent releases but few downloads"
  - "This crate has not been updated in the last 6 months and has few downloads"

For the crates used in the survey, assuming that any release in the last 6 mo
makes a crate "Experimental" rather than "Inactive", and that whatever the
cutoff value for "many downloads" is exactly, the line lies somewhere between
nom and combine since nom has an order of magnitude more downloads than combine:

| Crate   | Releases in last 6 mo | Downloads in last 90 days | Label        |
|---------|-----------------------|---------------------------|--------------|
| nom     | 3                     | 82,975                    | Mainstream   |
| combine | 4                     | 4,252                     | Experimental |
| lalrpop | 3                     | 1,928                     | Experimental |
| peg     | 7                     | 2,190                     | Experimental |
| peresil | 0                     | 1,859                     | Inactive     |

### Overall ordering: Recent downloads

To remove some of the bias towards older crates that may have been replaced with
newer alternatives, we propose that the default ranking of crates be changed
from the all-time number of downloads to the number of downloads in the last 90
days. This is easy to understand and explain, and is being used as a rough
measure of evaluation today. This should be enough to get the most suitable
crates on the first page of results.

## Out of scope

This proposal is not advocating to change the order of **search results**; those
should still be ordered by relevancy to the query based on the indexed content.
We may want to have an option to sort search results by "recommended" or
whatever we want to call this sorting, but probably not change the default.

# How do we teach this?

A criticism we anticipate and that would be totally fair is that this formula
is too complex. If we go with this formula, we think it's important to make
available a clear explanation of why a crate has the score it does, for
transparency to both crate users and crate authors. [Ruby toolbox][ruby] has a
great example of what we'd like to provide.

[ruby]: #ruby-toolbox

A possible benefit of having multiple measures influence the ranking is making
it less likely that crate owners will go to the effort of gaming the formula in
order to have a higher ranking.

# Drawbacks
[drawbacks]: #drawbacks

We might create a system that incentivizes attributes that are not useful, or
worse, actively harmful to the Rust ecosystem. For example, the documentation
percentage could be gamed by having one line of uninformative documentation for
all public items, thus giving a score of 100% without the value that would come
with a fully documented library. We hope the community at large will agree
these attributes are valuable to approach in good faith, and that trying to
game the ranking will be easily discoverable. We could have a reporting
mechanism for crates that are attempting to inflate their ranking artificially,
and implement a way for administrators to impose a ranking penalty on these
crates instead.

# Alternatives
[alternatives]: #alternatives

## Manual curation

1. We could keep the default ranking as number of downloads, and leave further
curation to sites like [Awesome Rust][].

[Awesome Rust]: https://github.com/kud1ing/awesome-rust

2. We could build entirely manual ranking into crates.io, as [Ember Observer][]
does. This would be a lot of work that would need to be done by someone, but
would presumably result in higher quality evaluations and be less vulnerable to
gaming.

[Ember Observer]: https://emberobserver.com/about

3. We could add user ratings or reviews in the form of upvote/downvote, 1-5
stars, and/or free text, and weight more recent ratings higher than older
ratings. This could have the usual problems that come with online rating
systems, such as spam, paid reviews, ratings influenced by personal
disagreements, etc.

## More options instead of a default

1. We could add filtering options for metadata, so that each user could choose,
for example, "show me only crates that work on stable" or "show me only crates
that have a version greater than 1.0".

2. We could add independent axes of sorting criteria in addition to the existing
alphabetical and number of downloads, such as by number of owners or most
recent version release date.

These sorting and filtering options would let each user choose exactly what's
important to them, which gives them more freedom, but this also pushes more
work onto the user. Crates.io would avoid taking a position on what "best"
means, which could prevent gaming of the system since crate authors wouldn't
know how users are ultimately sorting and filtering. We would probably want to
implement saved search configurations per user, so that people wouldn't have to
re-enter their criteria every time they wanted to do a similar search.

# Unresolved questions
[unresolved]: #unresolved-questions

- There might be metadata about crates that we haven't thought of yet that would
be useful.
- How do we change the ranking if we try something for a while and decide it's
not what we want? Would we need another RFC?
- How will we know this algorithm is working?
  - We could do another survey
  - We could ask for reports on an issue on crates.io of crates not being
    ordered as people would expect
  - Crates.io does have Google Analytics. We could compare the "funnels" of
    navigating to crate pages after searches that are similar to categories.
    This could potentially tell us if people start using categories at all
    instead of searching, if searches for terms that have categories go down
    and use of the categories go up. It might also be possible to see what
    crate pages people end up on from search and from categories, to see if
    they end up on "better" crates as a result of the ordering in categories.
    It might be difficult to get the right data in a significant quantity for
    this to be useful, though.
  - We could wait and see if there are complaints on the various Rust forums

# Appendix: Comparative Research
[comparative-research]: #appendix-comparative-research

This is how other package hosting websites handle default sorting within
categories.

## Django Packages

[Django Packages][django] has the concept of [grids][], which are large tables
of packages in a particular category. Each package is a column, and each row is
some attribute of packages. The default ordering from left to right appears to
be GitHub stars.

[django]: https://djangopackages.org/
[grids]: https://djangopackages.org/grids/

<img src="http://i.imgur.com/YAp9WYf.png" alt="Example of a Django Packages grid" width="800" />

## Libhunt

[Libhunt][libhunt] pulls libraries and categories from [Awesome Rust][], then
adds some metadata and navigation.

The default ranking is relative popularity, measured by GitHub stars and scaled
to be a number out of 10 as compared to the most popular crate. The other
ordering offered is dev activity, which again is a score out of 10, relative to
all other crates, and calculated by giving a higher weight to more recent
commits.

[libhunt]: https://rust.libhunt.com/

<img src="http://i.imgur.com/Yv6diFU.png" alt="Example of a Libhunt category" width="800" />

You can also choose to compare two libraries on a number of attributes:

<img src="http://i.imgur.com/HBtCH2E.png" alt="Example of comparing two crates on Libhunt" width="800" />

## Maven Repository

[Maven Repository][mvn] appears to order by the number of reverse dependencies
("# usages"):

[mvn]: http://mvnrepository.com

<img src="http://i.imgur.com/nZEQdAr.png" alt="Example of a maven repository category" width="800" />

## Pypi

[Pypi][pypi] lets you choose multiple categories, which are not only based on
topic but also other attributes like library stability and operating system:

[pypi]: https://pypi.python.org/pypi?%3Aaction=browse

<img src="http://i.imgur.com/Y3llc5m.png" alt="Example of filtering by Pypi categories" width="800" />

Once you've selected categories and click the "show all" packages in these
categories link, the packages are in alphabetical order... but the alphabet
starts over multiple times... it's unclear from the interface why this is the
case.

<img src="http://i.imgur.com/xEKGTsQ.jpg" alt="Example of Pypi ordering" width="800" />

## GitHub Showcases

To get incredibly meta, GitHub has the concept of [showcases][] for a variety
of topics, and they have [a showcase of package managers][show-pkg]. The
default ranking is by GitHub stars (cargo is 17/27 currently).

[showcases]: https://github.com/showcases
[show-pkg]: https://github.com/showcases/package-managers

<img src="http://i.imgur.com/SCvKQi2.png" alt="Example of a GitHub showcase" width="800" />

## Ruby toolbox

[Ruby toolbox][rb] sorts by a relative popularity score, which is calculated
from a combination of GitHub stars/watchers and number of downloads:

[rb]: https://www.ruby-toolbox.com

<img src="http://i.imgur.com/5Qt03n3.png" alt="How Ruby Toolbox's popularity ranking is calculated" width="800" />

Category pages have a bar graph showing the top gems in that category, which
looks like a really useful way to quickly see the differences in relative
popularity. For example, this shows nokogiri is far and away the most popular
HTML parser:

<img src="http://i.imgur.com/tj8emlu.png" alt="Example of Ruby Toolbox ordering" width="800" />

Also of note is the amount of information shown by default, but with a
magnifying glass icon that, on hover or tap, reveals more information without a
page load/reload:

<img src="http://i.imgur.com/0NPi6ct.png" alt="Expanded Ruby Toolbox info" width="800" />

## npms

While [npms][] doesn't have categories, its search appears to do some exact
matching of the query and then rank the rest of the results [weighted][] by
three different scores:

* score-effect:14: Set the effect that package scores have for the final search
  score, defaults to 15.3
* quality-weight:1: Set the weight that quality has for the each package score,
  defaults to 1.95
* popularity-weight:1: Set the weight that popularity has for the each package
  score, defaults to 3.3
* maintenance-weight:1: Set the weight that the quality has for the each
  package score, defaults to 2.05

[npms]: https://npms.io
[weighted]: https://api-docs.npms.io/

<img src="http://i.imgur.com/aWMeNv5.png" alt="Example npms search results" width="800" />

There are [many factors][] that go into the three scores, and more are planned
to be added in the future. Implementation details are available in the
[architecture documentation][].

[many factors]: https://npms.io/about
[architecture documentation]: https://github.com/npms-io/npms-analyzer/blob/master/docs/architecture.md

<img src="http://i.imgur.com/0i897ts.png" alt="Explanation of the data analyzed by npms" width="800" />

## Package Control (Sublime)

[Package Control][] is for Sublime Text packages. It has Labels that are
roughly equivalent to categories:

[Package Control]: https://packagecontrol.io/

<img src="http://i.imgur.com/81PGbFM.png" alt="Package Control homepage showing Labels like language syntax, snippets" width="800" />

The only available ordering within a label is alphabetical, but each result has
the number of downloads plus badges for Sublime Text version compatibility, OS
compatibility, Top 25/100, and new/trending:

<img src="http://i.imgur.com/KtWcOXV.png" alt="Sample Package Control list of packages within a label, sorted alphabetically" width="800" />

# Appendix: User Research
[user-research]: #appendix-user-research

## Demographics

We ran a survey for 1 week and got 134 responses. The responses we got seem to
be representative of the current Rust community: skewing heavily towards more
experienced programmers and just about evenly distributed between Rust
experience starting before 1.0, since 1.0, in the last year, and in the last 6
months, with a slight bias towards longer amounts of experience. 0 Graydons
responded to the survey.

<img src="http://i.imgur.com/huSYPyd.png" width="800" alt="Distribution of programming experience of survey repsondents, over half have been programming for over 10 years" />

<img src="http://i.imgur.com/t3kVXy9.png" width="800" alt="Distribution of Rust experience of survey respondents, slightly biased towards those who have been using Rust before 1.0 and since 1.0 over those with less than a year and less than 6 months" />

Since this matches about what we'd expect of the Rust community, we believe
this survey is representative. Given the bias towards more experience
programming, we think the answers are worthy of using to inform recommendations
crates.io will be making to programmers of all experience levels.

## Crate ranking agreement

The community ranking of the 5 crates presented in the survey for which order
people would try them out for parsing comes out to be:

1.) nom

2.) combine

3.) and 4.) peg and lalrpop, in some order

5.) peresil

This chart shows how many people ranked the crates in each slot:

<img src="http://i.imgur.com/x5SOTps.png" width="800" alt="Raw votes for each crate in each slot, showing that nom and combine are pretty clearly 1 and 2, peresil is clearly 5, and peg and lalrpop both got slotted in 4th most often" />

This chart shows the cumulative number of votes: each slot contains the number
of votes each crate got for that ranking or above.

<img src="http://i.imgur.com/QsfwVNj.png" width="800" alt="" />

Whatever default ranking formula we come up with in this RFC, when applied to
these 5 crates, it should generate an order for the crates that aligns with the
community ordering. Also, not everyone will agree with the crates.io ranking,
so we should display other information and provide alternate filtering and
sorting mechanisms so that people who prioritize different attributes than the
majority of the community will be able to find what they are looking for.

## Factors considered when ranking crates

The following table shows the top 25 mentioned factors for the two free answer
sections. We asked both "Please explain what information you used to evaluate
the crates and how that information influenced your ranking." and "Was there
any information you wish was available, or that would have taken more than 15
minutes for you to get?", but some of the same factors were deemed to take too
long to find out or not be easily available, while others did consider those,
so we've ranked by the combination of mentions of these factors in both
questions.

Far and away, good documentation was the most mentioned factor people used to
evaluate which crates to try.

|    | Feature                                                                        | Used in evaluation   | Not available/too much time needed | Total                     | Notes                 |
|----|--------------------------------------------------------------------------------|----------------------|------------------------------------|---------------------------|-----------------------|
| 1  | Good documentation                                                             | 94                   | 10                                 | 104                       |                       |
| 2  | README                                                                         | 42                   | 19                                 | 61                        |                       |
| 3  | Number of downloads                                                            | 58                   | 0                                  | 58                        |                       |
| 4  | Most recent version date                                                       | 54                   | 0                                  | 54                        |                       |
| 5  | Obvious / easy to find usage examples                                          | 37                   | 14                                 | 51                        |                       |
| 6  | Examples in the repo                                                           | 38                   | 6                                  | 44                        |                       |
| 7  | Reputation of the author                                                       | 36                   | 3                                  | 39                        |                       |
| 8  | Description or README containing Introduction / goals / value prop / use cases | 29                   | 5                                  | 34                        |                       |
| 9  | Number of reverse dependencies (Dependent Crates)                              | 23                   | 7                                  | 30                        |                       |
| 10 | Version >= 1.0.0                                                               | 30                   | 0                                  | 30                        |                       |
| 11 | Commit activity                                                                | 23                   | 6                                  | 29                        | Depends on VCS        |
| 12 | Fits use case                                                                  | 26                   | 3                                  | 29                        | Situational           |
| 13 | Number of dependencies (more = worse)                                          | 28                   | 0                                  | 28                        |                       |
| 14 | Number of open issues, activity on issues"                                     | 22                   | 6                                  | 28                        | Depends on GitHub     |
| 15 | Easy to use or understand                                                      | 27                   | 0                                  | 27                        | Situational           |
| 16 | Publicity (blog posts, reddit, urlo, "have I heard of it")                     | 25                   | 0                                  | 25                        |                       |
| 17 | Most recent commit date                                                        | 17                   | 5                                  | 22                        | Dependent on VCS      |
| 18 | Implementation details                                                         | 22                   | 0                                  | 22                        | Situational           |
| 19 | Nice API                                                                       | 22                   | 0                                  | 22                        | Situational           |
| 20 | Mentioned using/wanting to use docs.rs                                         | 8                    | 13                                 | 21                        |                       |
| 21 | Tutorials                                                                      | 18                   | 3                                  | 21                        |                       |
| 22 | Number or frequency of released versions                                       | 19                   | 1                                  | 20                        |                       |
| 23 | Number of maintainers/contributors                                             | 12                   | 6                                  | 18                        | Depends on VCS        |
| 24 | CI results                                                                     | 15                   | 2                                  | 17                        | Depends on CI service |
| 25 | Whether the crate works on nightly, stable, particular stable versions         | 8                    | 8                                  | 16                        |                       |

## Relevant quotes motivating our choice of factors

### Easy to use

> 1) Documentation linked from crates.io  2) Documentation contains decent
> example on front page

-----

> 3. "Docs Coverage" info - I'm not sure if there's a way to get that right
> now, but this is almost more important that test coverage.

-----

> rust docs:  Is there an intro and example on the top-level page?  are the
> rustdoc examples detailed enough to cover a range of usecases?  can i avoid
> reading through the files in the examples folder?

-----

> Documentation:
> - Is there a README? Does it give me example usage of the library? Point me
>   to more details?
> - Are functions themselves documented?
> - Does the documentation appear to be up to date?

-----

> The GitHub repository pages, because there are no examples or detailed
> descriptions on crates.io. From the GitHub readme I first checked the readme
> itself for a code example, to get a feeling for the library. Then I looked
> for links to documentation or tutorials and examples. The crates that did not
> have this I discarded immediately.

-----

> When evaluating any library from crates.io, I first follow the repository
> link -- often the readme is enough to know whether or not I like the actual
> library structure. For me personally a library's usability is much more
> important than performance concerns, so I look for code samples that show me
> how the library is used.    In the examples given, only peresil forces me to
> look at the actual documentation to find an example of use. I want something
> more than "check the docs" in a readme in regards to getting started.

-----

> I would like the entire README.md of each package to be visible on crates.io
> I would like a culture where each README.md contains a runnable example

-----

Ok, this one isn't from the survey, it's from [a Sept 2015 internals thread][]:

[a Sept 2015 internals thread]: https://users.rust-lang.org/t/lets-talk-about-ecosystem-documentation/2791/24?u=carols10cents

>> there should be indicator in Crates.io that show how much code is
>> documented, this would help with choosing well done package.
>
> I really love this idea! Showing a percentage or a little progress bar next
> to each crate with the proportion of public items with at least some docs
> would be a great starting point.

### Maintenance

> On nom's crates.io page I checked the version (2.0.0) and when the latest
> version came out (less than a month ago). I know that versioning is
> inconsistent across crates, but I'm reassured when a crate has V >= 1.0
> because it typically indicates that the authors are confident the crate is
> production-ready. I also like to see multiple, relatively-recent releases
> because it signals the authors are serious about maintenance.

-----

> Answering yes scores points:  crates.io page:  Does the crate have a major
> version >= 1?  Has there been a release recently, and maybe even a steady
> stream of minor or patch-level releases?

-----

> From github:
> * Number of commits and of contributors (A small number of commits (< 100)
> and of contributors (< 3) is often the sign of a personal project, probably
> not very much used except by its author. All other things equal, I tend to
> prefer active projects.);


### Quality

> Tests:
> - Is critical functionality well tested?
> - Is the entire package well tested?
> - Are the tests clear and descriptive?
> - Could I reimplement the library based on these tests?
> - Does the project have CI?
> - Is master green?

### Popularity/credibility

> 2) I look  at the number of download. If it is too small (~ <1000), I assume
> the crate has not yet reached a good quality. nom catches my attention
> because it has 200K download: I assume it is a high quality crate.

-----

> 1. Compare the number of downloads: More downloads = more popular = should be
> the best

-----

> Popularity:  - Although not being a huge factor, it can help tip the scale
> when one is more popular or well supported than another when all other
> factors are close.

### Overall

> I can't pick a most important trait because certain ones outweigh others when
> combined, etc. I.e. number of downloads is OK, but may only suggest that it's
> been around the longest. Same with number of dependent crates (which probably
> spikes number of downloads). I like a crate that is well documented, has a
> large user base (# dependent crates + downloads + stars), is post 1.0, is
> active (i.e. a release within the past 6 months?), and it helps when it's a
> prominent author (but that I feel is an unfair metric).

## Relevant bugs capturing other feedback

There was a wealth of good ideas and feedback in the survey answers, but not
all of it pertained to crate ranking directly. Commonly mentioned improvements
that could greatly help the usability and usefulness of crates.io included:

* [Rendering the README on crates.io](https://github.com/rust-lang/crates.io/issues/81)
* [Linking to docs.rs if the crate hasn't specified a Documentation link](https://github.com/rust-lang/crates.io/pull/459)
* [`cargo doc` should render crate examples and link to them on main documentation page](https://github.com/rust-lang/cargo/issues/2760)
* [`cargo doc` could support building/testing standalone markdown files](https://github.com/rust-lang/cargo/issues/739)
* [Allow documentation to be read from an external file](https://github.com/rust-lang/rust/issues/15470)
* [Have "favorite authors" and highlight crates by your favorite authors in crate lists](https://github.com/rust-lang/crates.io/issues/494)
* [Show the number of reverse dependencies next to the link](https://github.com/rust-lang/crates.io/issues/496)
* [Reverse dependencies should be ordered by number of downloads by default](https://github.com/rust-lang/crates.io/issues/495)
