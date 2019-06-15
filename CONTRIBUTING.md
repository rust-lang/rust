# Contributing to semverver

Want to help developing semverver? Cool! See below on how to do that.

## Bug reports

If you encounter any unwanted behaviour from the tool, such as crashes or other unexpected
output, or if you think you have found a bug, please open an issue in GitHub's issue
tracker.

Please describe the steps to reproduce your problem, as well as what you expected to
happen, and what happened instead. It is also useful to include the output of your command
invocation(s) with the following environment: `RUST_LOG=debug RUST_BACKTRACE=full`.
Please paste it at https://gist.github.com if the output is longer than a 50 lines or so.

## Feature requests

If you want to see some functionality added to semverver, you can also open an issue. Make
sure to include what functionality you need exactly, why it is useful, and, depending on
the complexity and/or how related the functionality is to the core project goals, why you
think it should be implemented in semverver and not somewhere else.

## Working on issues

If you want to write code to make semverver better, please post in the issue(s) you want
to tackle, and if you already have an idea/proposed solution, you are welcome to summarize
it in case some discussion is necessary.

Here are some guidelines you should try to stick to:

* Please fork the repository on GitHub and create a feature branch in your fork.
* Try to keep your code stylistically similar to the already existing codebase.
* Commit your changes in compact, logically coupled chunks.
* Make sure `cargo test` passes after your changes.
* Run `rustfmt` on your code (for example by running `cargo fmt`).
* If possible, fix any issues `cargo clippy` might find in your code.
* Finally, make a pull request against the master branch on GitHub and wait for the CI to
  find any outstanding issues.
