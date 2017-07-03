- Feature Name: tiered_browser_support
- Start Date: 2017-04-25
- RFC PR: https://github.com/rust-lang/rfcs/pull/1985
- Rust Issue: https://github.com/rust-lang/rust/issues/43035

# Summary
[summary]: #summary

Official web content produced by the Rust teams for consumption by Rust users
should work in the majority of browsers that Rust users are visiting these
sites in. The Rust compiler only supports [a finite number of targets][forge],
with varying degrees of support, due to the limits on time, expertise, and
testing resources. Similarly, we don't have enough time, expertise and testing
resources to be sure that our web content works in every version of every
browser. We should have a list of browsers and versions in various tiers of
support.

[forge]: https://forge.rust-lang.org/platform-support.html

# Motivation
[motivation]: #motivation

[This pull request to remove JQuery from rustdoc's output][jquery-pr] had
discussion about what we could and could not do because of browser support.
This is a discussion we haven't yet had as a community.

[jquery-pr]: https://github.com/rust-lang/rust/pull/41307

Crates.io doesn't display correctly in browsers without support for flexbox,
such as [Windows Phone 8.1][win-phone], a device that is no longer supported. I
made the decision that it wasn't worth it for the community to spend time
fixing this issue, did I make the correct tradeoff for the community?

[win-phone]: https://github.com/rust-lang/crates.io/issues/56

Supporting all versions of all browsers with the same behavior is impossible
with the small number of people who work on Rust's web content. Crates.io is
not currently doing any cross-browser testing; there are some JavaScript tests
of the UI that run in [PhantomJS][], a headless WebKit. Since we're not
testing, we don't actually know what our current web support even is, except
for when we get bug reports from users.

[PhantomJS]: http://phantomjs.org/

In order to fully test on all browsers to be sure of our support, we would
either need to have all the devices, operating systems, browsers, and versions
available and people with the time and inclination to do manual testing on all
of these, or we would need to be running automated tests on something like
[BrowserStack][]. BrowserStack does appear to have a free plan for open source
projects, but it's unclear how many parallel tests the open source plan would
give us, and we'd at least be spending time waiting for test results on the
various stacks. [BrowserStack also doesn't support every platform][bs-support],
Linux on the desktop being a notable section of our userbase missing from their
platforms.

[BrowserStack]: https://www.browserstack.com/pricing
[bs-support]: https://www.browserstack.com/support

# Detailed design
[design]: #detailed-design

## Rust web content

Officially produced web content includes:

- rust-lang.org
- blog.rust-lang.org
- play.rust-lang.org
- crates.io
- Rustdoc output
- thanks.rust-lang.org

Explicitly not included:

- Content for people working on Rust itself, such as:
  - [The Rust Forge][]
  - [rusty-dash][]

[The Rust Forge]: https://forge.rust-lang.org/
[rusty-dash]: https://rusty-dash.com/

Things that are not really under our control but are used for official or
almost-official Rust web content:

- GitHub
- docs.rs
- Discourse (used for [urlo][] and [irlo][])
- [mdBook][] output (used for the books and other documentation)

[urlo]: https://users.rust-lang.org/
[irlo]: https://internals.rust-lang.org/
[mdBook]: https://github.com/azerupi/mdBook/

## Proposed browser support tiers

Based on [actual usage metrics][] and with a goal of spending our time in an
effective way, the browser support tiers would be defined as:

[actual usage metrics]: #google-analytics-browser-usage-stats

Browsers are listed in [browserslist][] format.

[browserslist]: https://github.com/ai/browserslist

### Supported browsers

Goal: Ensure functionality of our web content for 80% of users.

Browsers:

```
last 2 Chrome versions
last 1 Firefox version
Firefox ESR
last 1 Safari version
last 1 iOS version
last 1 Edge version
last 1 UCAndroid version
```

[On browserl.ist](http://browserl.ist/?q=last+2+Chrome+versions%2C+last+1+Firefox+version%2C+Firefox+ESR%2C+last+1+Safari+version%2C+last+1+iOS+version%2C+last+1+Edge+version%2C+last+1+UCAndroid+version)

Support:

- We add automated testing of functionality in a variety of browsers through a
  service such as [BrowserStack][] for each of these as much as possible (and
  work on adding this type of automated testing to those web contents that
  aren't currently tested, such as rustdoc output).
- Bugs affecting the functionality of the sites in these browsers are
  prioritized highly.

### Unsupported browsers

Goal: Avoid spending large amounts of time and code complexity debugging and
hacking around quirks in older or more obscure browsers.

Browsers:

- Any not mentioned above

Support:

- No automated testing for these.
- Bug reports for these browsers are closed as WONTFIX.
- Pull requests to fix functionality for these browsers are accepted only if
  they're deemed to not add an inordinate amount of complexity or maintenance
  burden (subjective, reviewers' judgment).

## Orthogonal but related non-proposals

The following principles are assumptions I'm making that we currently follow
and that we should continue to strive for, no matter what browser support
policy we end up with:

- Follow best practices for accessibilty, fix bug reports from blind users,
  reach out to blind users in the community about how the accessibility of the
  web content could be improved.
  - This would include supporting lynx/links as these are sometimes used with
    screen readers.
- Follow best practices for colorblindness, such as have information conveyed
  through color also conveyed through an icon or text.
- Follow best practices for making content usable from mobile devices with a
  variety of screen sizes.
- Render content without requiring JavaScript (especially on
  [crates.io][noscript]). Additional functionality beyond reading (ex: search,
  follow/unfollow crate) may require JavaScript, but we will attempt to use
  links and forms for progressive enhancement as much as possible.

[noscript]: https://github.com/rust-lang/crates.io/issues/204

Please comment if you think any of these should **not** be assumed, but rest
assured it is not the intent of this RFC to get rid of these kinds of support.

# Relevant data

[CanIUse.com][] has some statistics on global usage of browsers and versions,
but our audience (developers) isn't the same as the general public.

[CanIUse.com]: http://caniuse.com/usage-table

## Google analytics browser usage stats

We have Google Analytics on crates.io and on rust-lang.org. The entire data set
of the usage stats by browser, browser verison, and OS are available [in this
Google sheet][all-data] for the visits to crates.io in the last month. I chose
to use just crates.io because on initial analysis, the top 90% of visits to
rust-lang.org were less varied than the top 90% of visits to crates.io.

[all-data]: https://docs.google.com/spreadsheets/d/1qgszm-_-Tn8FLi2v3vicuvyct3Grzz74JWcroILRq8s/edit?usp=sharing

This data does not include those users who block Google Analytics.

This is the top 80% aggregated by browser and major browser version:

| Browser         | Browser Version | Sessions | % of sessions | Cumulative % |
|-----------------|-----------------|----------|---------------|--------------|
| Chrome          | 57              | 18,040   | 34.85         | 34.85        |
| Firefox         | 52              | 8,136    | 15.72         | 50.56        |
| Chrome          | 56              | 7302     | 14.11         | 64.67        |
| Safari          | 10.1 (macos)    | 1,592    | 3.08          | 67.74        |
| Safari          | 10 (ios)        | 1,437    | 2.78          | 70.52        |
| Safari          | 10.0.3 (macos)  | 851      | 1.64          | 72.16        |
| Firefox         | 53              | 767      | 1.48          | 73.65        |
| Chrome          | 55              | 717      | 1.39          | 75.03        |
| Firefox         | 45              | 693      | 1.34          | 76.37        |
| UC Browser      | 11              | 530      | 1.02          | 77.40        |
| Chrome          | 58              | 520      | 1.00          | 78.40        |
| Safari (in-app) | (not set) (ios) | 500      | 0.97          | 79.37        |
| Firefox         | 54              | 472      | 0.91          | 80.28        |

Interesting to note: Firefox 45 is the latest
[ESR](https://www.mozilla.org/en-US/firefox/organizations/all/) (Firefox 52
will also be an ESR but it was just released). Firefox 52 was the current major
version for most of this past month; I'm guessing the early adopters of 53 and
54 are likely Mozilla employees.

## What do other sites in our niche support?

- [GitHub][] - Current versions of Chrome, Firefox, Safari, Edge and IE 11.
  Best effort for Firefox ESR.
- [Discourse][] - Chrome 32+, Firefox 27+, Safari 6.1+, IE 11+, iPad 3+, iOS
  8+, Android 4.3+ (doesn't specify which browser on the devices, doesn't look
  like they've updated these numbers in a while)

[GitHub]: https://help.github.com/articles/supported-browsers/
[Discourse]: https://github.com/discourse/discourse#requirements

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

We should call this "Rust Browser Support", and we should have the tiers listed
on the Rust Forge in a similar way to the tiers of Rust platforms supported.

We should link to the tiered browser support page from places where Rust web
content is developed and on the [Rust FAQ][].

[Rust FAQ]: https://www.rust-lang.org/en-US/faq.html

# Drawbacks
[drawbacks]: #drawbacks

We exclude some people who are unwilling or unable to use a modern browser.

# Alternatives
[alternatives]: #alternatives

We could adopt the tiers proposed above but with different browser versions.

We could adopt the browsers proposed above but with different levels of support.

Other alternatives:

## Not have official browser support tiers (status quo)

By not creating offical levels of browser support, we will continue to have the
situation we have today: discussions and decisions are happening that affect
the level of support that Rust web content has in various browsers, but we
don't have any agreed-upon guidelines to guide these discussions and decisions.

We continue to not test in multiple browsers, instead relying on bug reports
from users. The people doing the work continue to decide on an ad-hoc basis
whether a fix is worth making or not.

## Support all browsers in all configurations

We could choose to attempt to support any version of any browser on any device,
testing with as much as we can. We would still have to rely on bug reports and
help from the community to test with some configurations, but we wouldn't close
any bug report or pull request due to the browser or version required to
reproduce it.

# Unresolved questions
[unresolved]: #unresolved-questions

- Am I missing any official web content that this policy should apply to?
- Is it possible to add browser tests to rustdoc or would that just make the
  current situation of long, flaky rustc builds worse?
