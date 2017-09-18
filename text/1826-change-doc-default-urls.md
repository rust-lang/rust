- Feature Name: N/A
- Start Date: 2016-12-22
- RFC PR: https://github.com/rust-lang/rfcs/pull/1826
- Rust Issue: https://github.com/rust-lang/rust/issues/44687

# Summary
[summary]: #summary

Change doc.rust-lang.org to redirect to the latest release instead of an alias
of stable.

Introduce a banner that contains a dropdown allowing users to switch between versions,
noting when a release is not the most current release.

# Motivation
[motivation]: #motivation

Today, if you hit https://doc.rust-lang.org/, you'll see the same thing as if
you hit https://doc.rust-lang.org/stable/. It does not redirect, but instead
displays the same documentation. This is suboptimal for multiple reasons:

* One of the oldest bugs open in Rust, from September 2013 (a four digit issue
  number!), is about the lack of `rel=canonical`, which means search results
  are being duplicated between `/` and `/stable`, at least ([issue link][9461])
* `/` not having any version info is a similar bug, stated in a different way,
  but still has the same problems. ([issue link][14466])
* We've attempted to change the URL structure of Rustdoc in the past, but it's
  caused many issues, which will be elaborated below. ([issue link][34271])

[9461]: http://github.com/rust-lang/rust/issues/9461
[14466]: https://github.com/rust-lang/rust/issues/14466
[34271]: https://github.com/rust-lang/rust/issues/34271

There's other issues that stem from this as well that haven't been filed as
issues. Two notable examples are:

* When we release the new book, links are going to break. This has multiple
  ways of being addressed, and so isn't a strong motivation, but fixing this
  issue would help out a lot.
* In order to keep links working, we modified rustdoc [to add redirects from
  the older format](https://github.com/rust-lang/rust/issues/35020). But this
  can lead to degenerate situations in certain crates. `libc`, one of the most
  important crates in Rust, and included in the official docs, [had their docs
  break](https://github.com/rust-lang/libc/pull/438) because so many extra
  files were generated that GitHub Pages refused to serve them any more.

From `#rust-internals` on 2016-12-22:

```text
18:19 <@brson> lots of libc docs
18:19 <@steveklabnik> :(
18:20 <@brson> 6k to document every C constant
```

Short URLs are nice to have, but they have an increasing maintenance cost
that's affecting other parts of the project in an adverse way.

The big underlying issue here is that people tend to link to `/`, because it's
what you get by default. By changing the default, people will link to the
specific version instead. This means that their links will not break, and will
allow us to update the URL structure of our documentation more freely.

# Detailed design
[design]: #detailed-design

https://doc.rust-lang.org/ will be updated to have a heading
with a drop-down that allows you to select between different versions of the docs. It
will also display a message when looking at older documentation.

https://doc.rust-lang.org/ should issue a redirect to https://doc.rust-lang.org/RELEASE,
where RELEASE is the latest stable release, like `1.14.0`.

The exact details will be worked out before this is 'stabilized' on doc.rust-lang.org;
only the general approach is presented in this RFC.

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

There's not a lot to teach; users end up on a different page than they used to.

# Drawbacks
[drawbacks]: #drawbacks

Losing short URLs is a drawback. This is outweighed by other considerations,
in my opinion, as the rest of the RFC shows.

# Alternatives
[alternatives]: #alternatives

We could make no changes. We've dealt with all of these problems so far, so
it's possible that we won't run into more issues in the future.

We could do work on the `rel=canonical` issue instead, which would solve this
in a different way. This doesn't totally solve all issues, however, only
the duplication issue.

We could redirect all URLs that don't start with a version prefix to redirect to
`/`, which would be an index page showing all of the various places to go. Right
now, it's unclear how many people even know that we host specific old versions,
or stuff like `/beta`.

# Unresolved questions
[unresolved]: #unresolved-questions

None.
