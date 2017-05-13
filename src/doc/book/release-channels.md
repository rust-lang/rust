% Release Channels

The Rust project uses a concept called ‘release channels’ to manage releases.
It’s important to understand this process to choose which version of Rust
your project should use.

# Overview

There are three channels for Rust releases:

* Nightly
* Beta
* Stable

New nightly releases are created once a day. Every six weeks, the latest
nightly release is promoted to ‘Beta’. At that point, it will only receive
patches to fix serious errors. Six weeks later, the beta is promoted to
‘Stable’, and becomes the next release of `1.x`.

This process happens in parallel. So every six weeks, on the same day,
nightly goes to beta, beta goes to stable. When `1.x` is released, at
the same time, `1.(x + 1)-beta` is released, and the nightly becomes the
first version of `1.(x + 2)-nightly`.

# Choosing a version

Generally speaking, unless you have a specific reason, you should be using the
stable release channel. These releases are intended for a general audience.

However, depending on your interest in Rust, you may choose to use nightly
instead. The basic tradeoff is this: in the nightly channel, you can use
unstable, new Rust features. However, unstable features are subject to change,
and so any new nightly release may break your code. If you use the stable
release, you cannot use experimental features, but the next release of Rust
will not cause significant issues through breaking changes.

# Helping the ecosystem through CI

What about beta? We encourage all Rust users who use the stable release channel
to also test against the beta channel in their continuous integration systems.
This will help alert the team in case there’s an accidental regression.

Additionally, testing against nightly can catch regressions even sooner, and so
if you don’t mind a third build, we’d appreciate testing against all channels.

As an example, many Rust programmers use [Travis](https://travis-ci.org/) to
test their crates, which is free for open source projects. Travis [supports
Rust directly][travis], and you can use a `.travis.yml` file like this to
test on all channels:

```yaml
language: rust
rust:
  - nightly
  - beta
  - stable

matrix:
  allow_failures:
    - rust: nightly
```

[travis]: http://docs.travis-ci.com/user/languages/rust/

With this configuration, Travis will test all three channels, but if something
breaks on nightly, it won’t fail your build. A similar configuration is
recommended for any CI system, check the documentation of the one you’re
using for more details.
