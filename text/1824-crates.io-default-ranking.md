- Feature Name: crates_io_default_ranking
- Start Date: 2016-12-19
- RFC PR: https://github.com/rust-lang/rfcs/pull/1824
- Rust Issue: https://github.com/rust-lang/rust/issues/41616

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

## Order by recent downloads

Through the iterations of this RFC, there was no consensus around a way to order
crates that would be useful, understandable, resistent to being gamed, and not
require work of curators, reviewers, or moderators. Furthermore, different
people in different situations may value different aspects of crates.

Instead of attempting to order crates as a majority of people would rank them,
we propose a coarser measure to expose the set of crates worthy of further
consideration on the first page of a category or keyword. At that point, the
person looking for a crate can use other indicators on the page to decide which
crates best meet their needs.

**The default ordering of crates within a keyword or category will be changed to
be the number of downloads in the last 90 days.**

While coarse, downloads show how many people or other crates have found this
crate to be worthy of using. By limiting to the last 90 days, crates that have
been around the longest won't have an advantage over new crates that might be
better. Crates that are lower in the "stack", such as `libc`, will always have a
higher number of downloads than those higher in the stack due to the number of
crates using a lower-level crate as a dependency. Within a category or keyword,
however, crates are likely to be from the same level of the stack and thus their
download numbers will be comparable.

Crates are currently ordered by all-time downloads and the sort option button
says "Downloads". We will:

- change the ordering to be downloads in the last 90 days
- change the number of downloads displayed with each crate to be those made in
  the last 90 days
- change the sort option button to say "Recent Downloads".

"All-time Downloads" could become another sort option in the menu, alongside
"Alphabetical".

## Add more badges, filters, and sorting options

Crates.io now has badges for master branch CI status, and [will soon have a
badge indicating the version(s) of Rust a particular version builds
successfully on][build-info].

[build-info]: https://github.com/rust-lang/crates.io/pull/540

To enable a person to narrow down relevant crates to find the one that will best
meet their needs, we will add more badges and indicators. **Badges will not
influence crate ordering**.

Some badges may require use of third-party services such as GitHub. We recognize
that not everyone uses these services, but note a specific badge is only one
factor that people can consider out of many.

Through [the survey we conducted][user-research], we found that when people
evaluate crates, they are primarily looking for signals of:

- Ease of use
- Maintenance
- Quality

Secondary signals that were used to infer the primary signals:

- Popularity (covered by the default ordering by recent downloads)
- Credibility

### Ease of use

By far, the most common attribute people said they considered in the survey was
whether a crate had good documentation. Frequently mentioned when discussing
documentation was the desire to quickly find an example of how to use the crate.

This would be addressed in two ways.

#### Render README on a crate's page

[Render README files on a crate's page on crates.io][render-readme] so that
people can quickly see for themselves the information that a crate author
chooses to make available in their README. We can nudge towards having an
example in the README by adding a template README that includes an Examples
section [in what `cargo new` generates][cargo-new].

[render-readme]: https://github.com/rust-lang/crates.io/issues/81
[cargo-new]: https://github.com/rust-lang/cargo/issues/3506

#### "Well Documented" badge

For each crate published, in a background job, unpack the crate files and
calculate the ratio of lines of documentation to lines of code as follows:

- Find the number of lines of documentation in Rust files:
  `grep -r "//[!/]" --binary-files=without-match --include=*.rs . | wc -l`
- Find the number of lines in the README file, if specified in Cargo.toml
- Find the number of lines in Rust files: `find . -name '*.rs' | xargs wc -l`

We would then add the lines in the README to the lines of documentation,
subtract the lines of documentation from the total lines of code, and divide
the lines of documentation by the lines of non-documentation in order to get
the ratio of documentation to code. Test code (and any documentation within
test code) *is* part of this calculation.

Any crate getting in the top 20% of all crates would get a badge saying "well
documented".

This measure is gameable if a crate adds many lines that match the
documentation regex but don't provide meaningful content, such as `/// lol`.
While this may be easy to implement, a person looking at the documentation for
a crate using this technique would immediately be able to see that the author
is trying to game the system and reject it. If this becomes a common problem,
we can re-evaluate this situation, but we believe the community of crate
authors genuinely want to provide great documentation to crate users. We want
to encourage and reward well-documented crates, and this outweighs the risk of
potential gaming of the system.

* combine:
  * 1,195 lines of documentation
  * 99 lines in README.md
  * 5,815 lines of Rust
  * (1195 + 99) / (5815 - 1195) = 1294/4620 = .28

* nom:
  * 2,263 lines of documentation
  * 372 lines in README.md
  * 15,661 lines of Rust
  * (2263 + 372) / (15661 - 2263) = 2635/13398 = .20

* peresil:
  * 159 lines of documentation
  * 20 lines in README.md
  * 1,341 lines of Rust
  * (159 + 20) / (1341 - 159) = 179/1182 = .15

* lalrpop: ([in the /lalrpop directory in the repo][lalrpop-repo])
  * 742 lines of documentation
  * 110 lines in ../README.md
  * 94,104 lines of Rust
  * (742 + 110) / (94104 - 742) = 852/93362 = .01

* peg:
  * 3 lines of documentation
  * no readme specified in Cargo.toml
  * 1,531 lines of Rust
  * (3 + 0) / (1531 - 3) = 3/1528 = .00

[lalrpop-repo]: https://github.com/nikomatsakis/lalrpop/tree/master/lalrpop

If we assume these are all the crates on crates.io for this example, then
combine is the top 20% and would get a badge.

### Maintenance

We will add a way for maintainers to communicate their intended level of
maintenance and support. We will add indicators of issues resolved from the
various code hosting services.

#### Self-reported maintenance intention

We will add an optional attribute to Cargo.toml that crate authors could use to
self-report their maintenance intentions. The valid values would be along the
lines of the following, and would influence the ranking in the order they're
presented:

<dl>
  <dt>Actively developed</dt>
  <dd>
    New features are being added and bugs are being fixed.
  </dd>

  <dt>Passively maintained</dt>
  <dd>
    There are no plans for new features, but the maintainer intends to respond
    to issues that get filed.
  </dd>

  <dt>As-is</dt>
  <dd>
    The crate is feature complete, the maintainer does not intend to continue
    working on it or providing support, but it works for the purposes it was
    designed for.
  </dd>

  <dt><i>none</i></dt>
  <dd>
    We display nothing. Since the maintainer has not chosen to specify their
    intentions, potential crate users will need to investigate on their own.
  </dd>

  <dt>Experimental</dt>
  <dd>
    The author wants to share it with the community but is not intending to meet
    anyone's particular use case.
  </dd>

  <dt>Looking for maintainer</dt>
  <dd>
    The current maintainer would like to transfer the crate to someone else.
  </dd>
</dl>

These would be displayed as badges on lists of crates.

These levels would not have any time commitments attached to them-- maintainers
who would like to batch changes into releases every 6 months could report
"actively developed" just as much as mantainers who like to release every 6
weeks. This would need to be clearly communicated to set crate user
expectations properly.

This is also inherently a crate author's statement of current intentions, which
may get out of sync with the reality of the crate's maintenance over time.

If I had to guess for the maintainers of the parsing crates, I would assume:

* nom: actively developed
* combine: actively developed
* lalrpop: actively developed
* peg: actively developed
* peresil: passively maintained

#### GitHub issue badges

[isitmaintained.com][] provides badges indicating the time to resolution of GitHub issues and percentage of GitHub issues that are open.

[isitmaintained.com]: http://isitmaintained.com/

We will enable maintainers to add these badges to their crate.

| Crate | Issue Resolution | Open Issues |
|-------|------------------|-------------|
| combine | [![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/Marwes/combine.svg)](http://isitmaintained.com/project/Marwes/combine "Average time to resolve an issue") | [![Percentage of issues still open](http://isitmaintained.com/badge/open/Marwes/combine.svg)](http://isitmaintained.com/project/Marwes/combine "Percentage of issues still open") |
| nom | [![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/Geal/nom.svg)](http://isitmaintained.com/project/Geal/nom "Average time to resolve an issue") | [![Percentage of issues still open](http://isitmaintained.com/badge/open/Geal/nom.svg)](http://isitmaintained.com/project/Geal/nom "Percentage of issues still open") |
| lalrpop | [![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/nikomatsakis/lalrpop.svg)](http://isitmaintained.com/project/nikomatsakis/lalrpop "Average time to resolve an issue") | [![Percentage of issues still open](http://isitmaintained.com/badge/open/nikomatsakis/lalrpop.svg)](http://isitmaintained.com/project/nikomatsakis/lalrpop "Percentage of issues still open") |
| peg | [![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/kevinmehall/rust-peg.svg)](http://isitmaintained.com/project/kevinmehall/rust-peg "Average time to resolve an issue") | [![Percentage of issues still open](http://isitmaintained.com/badge/open/kevinmehall/rust-peg.svg)](http://isitmaintained.com/project/kevinmehall/rust-peg "Percentage of issues still open") |
| peresil | [![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/shepmaster/peresil.svg)](http://isitmaintained.com/project/shepmaster/peresil "Average time to resolve an issue") | [![Percentage of issues still open](http://isitmaintained.com/badge/open/shepmaster/peresil.svg)](http://isitmaintained.com/project/shepmaster/peresil "Percentage of issues still open") |

### Quality

We will enable maintainers to add [Coveralls][] badges to indicate the
crate's test coverage. If there are other services offering test coverage
reporting and badges, we will add support for those as well, but this is the
only service we know of at this time that offers code coverage reporting that
works with Rust projects.

[Coveralls]: https://coveralls.io

This excludes projects that cannot use Coveralls, which only currently supports
repositories hosted on GitHub or BitBucket that use CI on Travis, CircleCI,
Jenkins, Semaphore, or Codeship.

nom has coveralls.io configured: [![Coverage Status](https://coveralls.io/repos/Geal/nom/badge.svg?branch=master)](https://coveralls.io/r/Geal/nom?branch=master)

### Credibility

We have [an idea for a "favorite authors" list][favs] that we
think would help indicate credibility. With this proposed feature, each person
can define "credibility" for themselves, which makes this measure less gameable
and less of a popularity contest.

[favs]: https://github.com/rust-lang/crates.io/issues/494

## Out of scope

This proposal is not advocating to change the default order of **search
results**; those should still be ordered by relevancy to the query based on the
indexed content. We will add the ability to sort search results by recent
downloads.

# Evaluation

If ordering by number of recent downloads and providing more indicators is not
helpful, we expect to get bug reports from the community and feedback on the
users forum, reddit, IRC, etc.

In the community survey scheduled to be taken around May 2017, we will ask
about people's satisfaction with the information that crates.io provides.

If changes are needed that are significant, we will open a new RFC. If smaller
tweaks need to be made, the process will be managed through crates.io's issues.
We will consult with the tools team and core team to determine whether a change
is significant enough to warrant a new RFC.

# How do we teach this?

We will change the label on the default ordering button to read "Recent
Downloads" rather than "Downloads".

Badges will have tooltips on hover that provide additional information.

We will also add a page to doc.crates.io that details all possible indicators
and their values, and explains to crate authors how to configure or earn the
different badges.

# Drawbacks
[drawbacks]: #drawbacks

We might create a system that incentivizes attributes that are not useful, or
worse, actively harmful to the Rust ecosystem. For example, the documentation
percentage could be gamed by having one line of uninformative documentation for
all public items, thus giving a score of 100% without the value that would come
with a fully documented library. We hope the community at large will agree
these attributes are valuable to approach in good faith, and that trying to
game the badges will be easily discoverable. We could have a reporting
mechanism for crates that are attempting to gain badges artificially, and
implement a way for administrators to remove badges from those crates.

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

## More sorting and filtering options

There are even more options for interacting with the metadata that crates.io
has than we are proposing in this RFC at this time. For example:

1. We could add filtering options for metadata, so that each user could choose,
for example, "show me only crates that work on stable" or "show me only crates
that have a version greater than 1.0".

2. We could add independent axes of sorting criteria in addition to the existing
alphabetical and number of downloads, such as by number of owners or most
recent version release date.

We would probably want to implement saved search configurations per user, so
that people wouldn't have to re-enter their criteria every time they wanted to
do a similar search.

# Unresolved questions
[unresolved]: #unresolved-questions

All questions have now been resolved.

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
