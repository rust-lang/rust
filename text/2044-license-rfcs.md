- Feature Name: license_rfcs
- Start Date: 2017-06-26
- RFC PR: https://github.com/rust-lang/rfcs/pull/2044
- Rust Issue: https://github.com/rust-lang/rust/issues/43461

# Summary
[summary]: #summary

Introduce a move to dual-MIT/Apache2 licensing terms to the Rust RFCs repo, by
requiring them for all new contributions, and asking previous contributors to
agree on the new license.

# Disclaimer
[disclaimer]: #disclaimer

This RFC is not authored by a lawyer, so its reasoning may be wrong.

# Motivation
[motivation]: #motivation

Currently, the Rust RFCs repo is in a state where no clear open source license
is specified.

The current legal base of the the RFCs repo is the "License Grant to Other
Users" from the [Github ToS]`*`:

```
Any Content you post publicly, including issues, comments, and contributions to other Users' repositories, may be viewed by others. By setting your repositories to be viewed publicly, you agree to allow others to view and "fork" your repositories (this means that others may make their own copies of your Content in repositories they control).

If you set your pages and repositories to be viewed publicly, you grant each User of GitHub a nonexclusive, worldwide license to access your Content through the GitHub Service, and to use, display and perform your Content, and to reproduce your Content solely on GitHub as permitted through GitHub's functionality.
```

These terms may be sufficient for display of the rfcs repository on Github, but
it limits contributions and use, and even poses a risk.

The Github ToS grant only applies towards reproductions through the Github
Service. Hypothetically, if the Github Service ceases at some point in the
future, without a legal successor offering a replacement service, the RFCs may
not be redistributed any more.

Second, there are companies which have set up policies that limit their
employees to contribute to the RFCs repo in this current state.

Third, there is the possibility that Rust may undergo standardisation and
produce a normative document describing the language.
Possibly, the authors of such a document may want to include text from RFCs.

Fourth, the spirit of the Rust project is to be open source, and the current
terms don't fulfill any popular open source definition.

`*`: The Github ToS is licensed under the [Creative Commons Attribution license](https://creativecommons.org/licenses/by/4.0/)

[Github ToS]: https://help.github.com/articles/github-terms-of-service/#5-license-grant-to-other-users

# Detailed design
[design]: #detailed-design

After this RFC has been merged, all new RFCs will be required to be
dual-licensed under the MIT/Apache2. This includes RFCs currently being
[considered for merging].

`README.md` should include a note that all contributions to the repo should be
licensed under the new terms.

As the licensing requires consent from the RFC creators, an issue will be
created on rust-lang/rfcs with a list of past contributors to the repo,
asking every contributor to agree to their contributions to be licensed under
those terms.

Regarding non-RFC files in this repo, the intention is to get them licensed
as well, not just the RFCs themselves. Therefore, contributors should be asked
to license *all* their contributions to this repo, not just to the RFC files,
and *all* new contributions to this repo should be required to be licensed
under the new terms.

[considered for merging]: https://github.com/rust-lang/rfcs/pulls

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

The issue created should @-mention all Github users who have contributed,
generating a notification for each past contributor.

Also, after this RFC got merged, all RFCs in the queue will get a comment in
their Github PR and be asked to include the copyright section at the top of
their RFC file.

The note in README.md should should inform new PR authors of the terms
they put their contribution under.

# Drawbacks
[drawbacks]: #drawbacks

This is additional churn and pings a bunch of people, which they may not like.

# Alternatives
[alternatives]: #alternatives

Other licenses more suited for text may have been chosen, like the CC-BY
license. However, RFCs regularly include code snippets, which may be used in
the rust-lang/rust, and similarly, RFCs may want to include code snippets from
rust-lang/rust. It might be the case that the CC-BY license allows such
sharing, but it might also mean complications.

Also, the [swift-evolution](https://github.com/apple/swift-evolution)
repository is put under the Apache license as well.

Maybe for something like this, no RFC is needed. However, there exists
precedent on non technical RFCs with RFC 1636. Also, this issue has been known
for years and no action has been done on this yet. If this RFC gets closed as
too trivial or offtopic, and the issue is being acted upon, its author
considers it a successful endeavor.

# Links to previous discussion

* https://github.com/rust-lang/rfcs/issues/1259
* https://github.com/rust-lang/rust/issues/25664
* https://internals.rust-lang.org/t/license-the-rfcs-repo-under-the-cc-by-4-0-license/3870

# Unresolved questions
[unresolved]: #unresolved-questions

Should trivial contributions that don't fall under copyright be special cased?
This is probably best decided on a case by case basis, and only after a
contributor has been unresponsive or has disagreed with the new licensing
terms.
