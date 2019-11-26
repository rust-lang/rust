# Diagnostics ICE-breakers

**Github Label:** [A-diagnostics]

[A-Diagnostics]: https://github.com/rust-lang/rust/labels/A-diagnostics

The "Diagnostics ICE-breakers" are focused on bugs that center around the
user visible compiler output. These bugs can fall under one of multiple topics:

- [D-papercut]: Errors that needs small tweaks to archieve a good output.
- [D-newcomer-roadblock]: Errors that are hard to understand for new users.
- [D-confusing]: Errors that are hard to understand, regardless of experience
level.
- [D-invalid-suggestion]: Structured suggestions use heuristics to suggest
valid code that captures the user's intent, but these heuristics can be wrong.
This label aggregates cases where the heuristics don't account for some case.
- [D-verbose]: Sometimes errors lean towards verbosity to try and increase
understandability, but in practice the "wall of text" effect can be counter
productive. Tickets labeled this way are about _removing_ output from existing
errors.
- [D-incorrect]: A diagnostic that is giving misleading or incorrect
information. This might require creating a new, more targetted, error.
- [D-inconsistent]: Inconsistency in formatting, grammar or style between
diagnostic messages. This is usually related to capitalization or sentence
tense.
- [D-edition]: error that should account for edition differences, but doesn't.
- [A-diagnostic-suggestions]: error that should have a structured suggestion,
but don't.


[D-papercut]: https://github.com/rust-lang/rust/labels/D-papercut
[D-newcomer-roadblock]: https://github.com/rust-lang/rust/labels/D-newcomer-roadblock
[D-confusing]: https://github.com/rust-lang/rust/labels/D-confusing
[D-invalid-suggestion]: https://github.com/rust-lang/rust/labels/D-invalid-suggestion
[D-verbose]: https://github.com/rust-lang/rust/labels/D-verbose
[D-incorrect]: https://github.com/rust-lang/rust/labels/D-incorrect
[D-inconsistent]: https://github.com/rust-lang/rust/labels/D-inconsistent
[D-edition]: https://github.com/rust-lang/rust/labels/D-edition
[A-diagnostic-suggestions]: https://github.com/rust-lang/rust/labels/A-diagnostic-suggestions

For more information, visit the [diagnostics page].

[diagnostics page]: ../diagnostics.md
