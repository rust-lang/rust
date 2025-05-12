# Crater

[Crater](https://github.com/rust-lang/crater) is a tool for compiling and
running tests for _every_ crate on [crates.io](https://crates.io) (and a few on
GitHub). It is mainly used for checking the extent of breakage when implementing
potentially breaking changes and ensuring lack of breakage by running beta vs
stable compiler versions.

## When to run Crater

You should request a crater run if your PR makes large changes to the compiler
or could cause breakage. If you are unsure, feel free to ask your PR's reviewer.

## Requesting Crater Runs

The rust team maintains a few machines that can be used for running crater runs
on the changes introduced by a PR. If your PR needs a crater run, leave a
comment for the triage team in the PR thread. Please inform the team whether you
require a "check-only" crater run, a "build only" crater run, or a
"build-and-test" crater run. The difference is primarily in time; the
conservative (if you're not sure) option is to go for the build-and-test run. If
making changes that will only have an effect at compile-time (e.g., implementing
a new trait) then you only need a check run.

Your PR will be enqueued by the triage team and the results will be posted when
they are ready. Check runs will take around ~3-4 days, with the other two taking
5-6 days on average.

While crater is really useful, it is also important to be aware of a few
caveats:

- Not all code is on crates.io! There is a lot of code in repos on GitHub and
  elsewhere. Also, companies may not wish to publish their code. Thus, a
  successful crater run is not a magically green light that there will be no
  breakage; you still need to be careful.

- Crater only runs Linux builds on x86_64. Thus, other architectures and
  platforms are not tested. Critically, this includes Windows.

- Many crates are not tested. This could be for a lot of reasons, including that
  the crate doesn't compile any more (e.g. used old nightly features), has
  broken or flaky tests, requires network access, or other reasons.

- Before crater can be run, `@bors try` needs to succeed in building artifacts.
  This means that if your code doesn't compile, you cannot run crater.
