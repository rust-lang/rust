# About this guide

This guide is meant to help document how rustc – the Rust compiler –
works, as well as to help new contributors get involved in rustc
development. It is not meant to replace code documentation – each
chapter gives only high-level details – the kinds of things that
(ideally) don't change frequently.

There are three parts to this guide. Part 1 contains information that should
be useful no matter how you are contributing. Part 2 contains information
about how the compiler works. Finally, there are some appendices at the
end with useful reference information.

The guide itself is of course open-source as well, and the sources can
be found at the [GitHub repository]. If you find any mistakes in the
guide, please file an issue about it, or even better, open a PR
with a correction!

## Other places to find information

You might also find the following sites useful:

- [Rustc API docs] -- rustdoc documentation for the compiler
- [Forge] -- contains documentation about rust infrastructure, team procedures, and more
- [compiler-team] -- the home-base for the rust compiler team, with description
  of the team procedures, active working groups, and the team calendar.

[GitHub repository]: https://github.com/rust-lang/rustc-guide/
[Rustc API docs]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/
[Forge]: https://forge.rust-lang.org/
[compiler-team]: https://github.com/rust-lang/compiler-team/
