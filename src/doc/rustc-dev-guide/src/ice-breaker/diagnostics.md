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

## Diagnostic output style guide

The main parts of a diagnostic error are the following:

```
error[E0000]: main error message
  --> file.rs:LL:CC
   |
LL | <code>
   | -^^^^- secondary label
   |  |
   |  primary label
   |
   = note: note without a `Span`, created with `.note`
note: sub-diagnostic message for `.span_note`
  --> file.rs:LL:CC
   |
LL | more code
   |      ^^^^
```

- Description (`error`, `warning`, etc.).
- Code (for example, for "mismatched types", it is `E0308`). It helps
  users get more information about the current error through an extended
  description of the problem in the error code index.
- Message. It is the main description of the problem. It should be general and
  able to stand on its own, so that it can make sense even in isolation.
- Diagnostic window. This contains several things:
  - The path, line number and column of the beginning of the primary span.
  - The users' affected code and its surroundings.
  - Primary and secondary spans underlying the users' code. These spans can
    optionally contain one or more labels.
    - Primary spans should have enough text to descrive the problem in such a
      way that if it where the only thing being displayed (for example, in an
      IDE) it would still make sense. Because it is "spatially aware" (it
      points at the code), it can generally be more succinct than the error
      message.
    - If cluttered output can be foreseen in cases when multiple span labels
      overlap, it is a good idea to tweak the output appropriately. For
      example, the `if/else arms have incompatible types` error uses different
      spans depending on whether the arms are all in the same line, if one of
      the arms is empty and if none of those cases applies.
- Sub-diagnostics. Any error can have multiple sub-diagnostics that look
  similar to the main part of the error. These are used for cases where the
  order of the explanation might not correspond with the order of the code. If
  the order of the explanation can be "order free", leveraging secondary labels
  in the main diagnostic is preferred, as it is typically less verbose.

The text should be matter of fact and avoid capitalization and periods, unless
multiple sentences are _needed_:

```
error: the fobrulator needs to be krontrificated
```

When code or an identifier must appear in an message or label, it should be
surrounded with single acute accents \`.

### Structured suggestions

Structured suggestions are a special kind of annotation in a diagnostic that
let third party tools (like `rustfix` and `rust-analyzer`) apply these changes
with no or minimal user interaction. These suggestions have a degree of
confidence in the suggested code, from high
(`Applicability::MachineApplicable`) to low (`Applicability::MaybeIncorrect`).
Be conservative when choosing the level.

They point to one or more spans with corresponding code that will replace their
current content.

The message that accompanies them should be understandable in the following
contexts:

- shown as an independent sug-diagnostic (this is the default output)
- shown as a label pointing at the affected span (this is done automatically if
the some heuristics for verbosity are met)
- shown as a `help` sub-diagnostic with no content (used for cases where the
suggestion is obvious from the text, but we still want to let tools to apply
them))
- not shown (used for _very_ obvious cases, but we still want to allow tools to
apply them)


## Helpful tips and options

### Finding the source of errors

There are two main ways to find where a given error is emitted:

- `grep` for either a sub-part of the error message/label or error code. This
  usually works well and is straightforward, but there are some cases where
  the error emitting code is removed from the code where the error is
  constructed behind a relatively deep call-stack. Even then, it is a good way
  to get your bearings.
- Invoking `rustc` with the nightly-only flag `-Ztreat-err-as-bug=1`, which
  will treat the first error being emitted as an Internal Compiler Error, which
  allows you to use the environment variable `RUST_BACKTRACE=full` to get a
  stack trace at the point the error has been emitted. Change the `1` to
  something else if you whish to trigger on a later error. Some limitations
  with this approach is that some calls get elided from the stack trace because
  they get inlined in the compiled `rustc`, and the same problem we faced with
  the prior approach, where the _construction_ of the error is far away from
  where it is _emitted_. In some cases we buffer multiple errors in order to
  emit them in order.

The regular development practices apply: judicious use of `debug!()` statements
and use of a debugger to trigger break points in order to figure out in what
order things are happening.
