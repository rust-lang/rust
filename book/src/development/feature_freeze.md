# IMPORTANT: FEATURE FREEZE

This is a temporary notice.

From the 26th of June until the 18th of September we will perform a feature freeze. Only bugfix PRs will be reviewed
except already open ones. Every feature-adding PR opened in between those dates will be moved into a
milestone to be reviewed separately at another time.

We do this because of the long backlog of bugs that need to be addressed
in order to continue being the state-of-the-art linter that Clippy has become known for being.

## For contributors

If you are a contributor or are planning to become one, **please do not open a lint-adding PR**, we have lots of open
bugs of all levels of difficulty that you can address instead!

We currently have about 800 lints, each one posing a maintainability challenge that needs to account to every possible
use case of the whole ecosystem. Bugs are natural in every software, but the Clippy team considers that Clippy needs a
refinement period.

If you open a PR at this time, we will not review it but push it into a milestone until the refinement period ends,
adding additional load into our reviewing schedules.

## I want to help, what can I do

Thanks a lot to everyone who wants to help Clippy become better software in this feature freeze period!
If you'd like to help, making a bugfix, making sure that it works, and opening a PR is a great step!

To find things to fix, go to the [tracking issue][tracking_issue], find an issue that you like, go there and claim that
issue with `@rustbot claim`.

As a general metric and always taking into account your skill and knowledge level, you can use this guide:

- 🟥 [ICEs][search_ice], these are compiler errors that causes Clippy to panic and crash. Usually involves high-level
debugging, sometimes interacting directly with the upstream compiler. Difficult to fix but a great challenge that
improves a lot developer workflows!

- 🟧 [Suggestion causes bug][sugg_causes_bug], Clippy suggested code that changed logic in some silent way.
Unacceptable, as this may have disastrous consequences. Easier to fix than ICEs

- 🟨 [Suggestion causes error][sugg_causes_error], Clippy suggested code snippet that caused a compiler error
when applied. We need to make sure that Clippy doesn't suggest using a variable twice at the same time or similar
easy-to-happen occurrences.

- 🟩 [False positives][false_positive], a lint should not have fired, the easiest of them all, as this is "just"
identifying the root of a false positive and making an exception for those cases.

Note that false negatives do not have priority unless the case is very clear, as they are a feature-request in a
trench coat.

[search_ice]: https://github.com/rust-lang/rust-clippy/issues?q=sort%3Aupdated-desc+state%3Aopen+label%3A%22I-ICE%22
[sugg_causes_bug]: https://github.com/rust-lang/rust-clippy/issues?q=sort%3Aupdated-desc%20state%3Aopen%20label%3AI-suggestion-causes-bug
[sugg_causes_error]: https://github.com/rust-lang/rust-clippy/issues?q=sort%3Aupdated-desc%20state%3Aopen%20label%3AI-suggestion-causes-error%20
[false_positive]: https://github.com/rust-lang/rust-clippy/issues?q=sort%3Aupdated-desc%20state%3Aopen%20label%3AI-false-positive
[tracking_issue]: https://github.com/rust-lang/rust-clippy/issues/15086
