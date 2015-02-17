# Contributing to Rust

Thank you for your interest in contributing to Rust! There are many ways to
contribute, and we appreciate all of them. This document is a bit long, so here's
links to the major sections:

* [Feature Requests](#feature-requests)
* [Bug Reports](#bug-reports)
* [Pull Requests](#pull-requests)
* [Writing Documentation](#writing-documentation)
* [Issue Triage](#issue-triage)
* [Out-of-tree Contributions](#out-of-tree-contributions)

If you have questions, please make a post on [internals.rust-lang.org][internals] or
hop on [#rust-internals][pound-rust-internals].

As a reminder, all contributors are expected to follow our [Code of Conduct](coc).

[pound-rust-internals]: http://chat.mibbit.com/?server=irc.mozilla.org&channel=%23rust-internals
[internals]: http://internals.rust-lang.org
[coc]: http://www.rust-lang.org/conduct.html

## Feature Requests

To request a change to the way that the Rust language works, please open an
issue in the [RFCs repository](https://github.com/rust-lang/rfcs/issues/new)
rather than this one. New features and other significant language changes
must go through the RFC process.

## Bug Reports

While bugs are unfortunate, they're a reality in software. We can't fix what we
don't know about, so please report liberally. If you're not sure if something
is a bug or not, feel free to file a bug anyway.

If you have the chance, before reporting a bug, please [search existing
issues](https://github.com/rust-lang/rust/search?q=&type=Issues&utf8=%E2%9C%93),
as it's possible that someone else has already reported your error. This doesn't
always work, and sometimes it's hard to know what to search for, so consider this
extra credit. We won't mind if you accidentally file a duplicate report.

Opening an issue is as easy as following [this
link](https://github.com/rust-lang/rust/issues/new) and filling out the fields.
Here's a template that you can use to file a bug, though it's not necessary to
use it exactly:

    <short summary of the bug>

    I tried this code:

    <code sample that causes the bug>

    I expected to see this happen: <explanation>

    Instead, this happened: <explanation>

    ## Meta

    `rustc --version --verbose`:

    Backtrace:

All three components are important: what you did, what you expected, what
happened instead. Please include the output of `rustc --version --verbose`,
which includes important information about what platform you're on, what
version of Rust you're using, etc.

Sometimes, a backtrace is helpful, and so including that is nice. To get
a backtrace, set the `RUST_BACKTRACE` environment variable. The easiest way
to do this is to invoke `rustc` like this:

```bash
$ RUST_BACKTRACE=1 rustc ...
```

## Pull Requests

Pull requests are the primary mechanism we use to change Rust. GitHub itself
has some [great documentation][pull-requests] on using the Pull Request
feature. We use the 'fork and pull' model described there.

[pull-requests]: https://help.github.com/articles/using-pull-requests/

Please make pull requests against the `master` branch.

All pull requests are reviewed by another person. We have a bot,
@rust-highfive, that will automatically assign a random person to review your request.

If you want to request that a specific person reviews your pull request,
you can add an `r?` to the message. For example, Steve usually reviews
documentation changes. So if you were to make a documentation change, add

    r? @steveklabnik

to the end of the message, and @rust-highfive will assign @steveklabnik instead
of a random person. This is entirely optional.

After someone has reviewed your pull request, they will leave an annotation
on the pull request with an `r+`. It will look something like this:

    @bors: r+ 38fe8d2

This tells @bors, our lovable integration bot, that your pull request has
been approved. The PR then enters the [merge queue][merge-queue], where @bors
will run all the tests on every platform we support. If it all works out,
@bors will merge your code into `master` and close the pull request.

[merge-queue]: http://buildbot.rust-lang.org/homu/queue/rust

## Writing Documentation

Documentation improvements are very welcome. The source of `doc.rust-lang.org`
is located in `src/doc` in the tree, and standard API documentation is generated
from the source code itself.

Documentation pull requests function in the same as other pull requests, though
you may see a slightly different form of `r+`:

    @bors: r+ 38fe8d2 rollup

That additional `rollup` tells @bors that this change is eligible for a 'rollup'.
To save @bors some work, and to get small changes through more quickly, when
@bors attempts to merge a commit that's rollup-eligible, it will also merge
the other rollup-eligible patches too, and they'll get tested and merged at
the same time.

## Issue Triage

Sometimes, an issue will stay open, even though the bug has been fixed. And
sometimes, the original bug may go stale because something has changed in the
meantime.

It can be helpful to go through older bug reports and make sure that they are
still valid. Load up an older issue, double check that it's still true, and
leave a comment letting us know if it is or is not. The [least recently updated sort][lru] is good for finding issues like this.

[lru]: https://github.com/rust-lang/rust/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-asc

## Out-of-tree Contributions

There are a number of other ways to contribute to Rust that don't deal with
this repository.

Answer questions in [#rust][pound-rust], or on [users.rust-lang.org][users],
or on [StackOverflow][so].

Participate in the [RFC process](https://github.com/rust-lang/rfcs).

Find a [requested community library][community-library], build it, and publish
it to [Crates.io](http://crates.io). Easier said than done, but very, very
valuable!

[pound-rust]: http://chat.mibbit.com/?server=irc.mozilla.org&channel=%23rust
[users]: http://users.rust-lang.org/
[so]: http://stackoverflow.com/questions/tagged/rust
[community-library]: https://github.com/rust-lang/rfcs/labels/A-community-library
