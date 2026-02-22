# Translation

<div class="warning">
rustc's current diagnostics translation infrastructure (as of
<!-- date-check --> October 2024
) unfortunately causes some friction for compiler contributors, and the current
infrastructure is mostly pending a redesign that better addresses needs of both
compiler contributors and translation teams. Note that there is no current
active redesign proposals (as of
<!-- date-check --> October 2024
)!

Please see the tracking issue <https://github.com/rust-lang/rust/issues/132181>
for status updates.

The translation infra is waiting for a yet-to-be-proposed redesign and thus rework, we are not
mandating usage of current translation infra. Use the infra if you *want to* or
otherwise makes the code cleaner, but otherwise sidestep the translation infra
if you need more flexibility.
</div>

rustc's diagnostic infrastructure supports translatable diagnostics using
[Fluent].

## Writing translatable diagnostics

There are two ways of writing translatable diagnostics:

1. For simple diagnostics, using a diagnostic (or subdiagnostic) derive.
   ("Simple" diagnostics being those that don't require a lot of logic in
   deciding to emit subdiagnostics and can therefore be represented as
   diagnostic structs). See [the diagnostic and subdiagnostic structs
   documentation](./diagnostic-structs.md).
2. Using typed identifiers with `Diag` APIs (in
   `Diagnostic` or `Subdiagnostic` or `LintDiagnostic` implementations).

When adding or changing a translatable diagnostic,
you don't need to worry about the translations.
Only updating the original English message is required.

## Fluent

Fluent is built around the idea of "asymmetric localization", which aims to
decouple the expressiveness of translations from the grammar of the source
language (English in rustc's case). Prior to translation, rustc's diagnostics
relied heavily on interpolation to build the messages shown to the users.
Interpolated strings are hard to translate because writing a natural-sounding
translation might require more, less, or just different interpolation than the
English string, all of which would require changes to the compiler's source
code to support.

Diagnostic messages are defined in Fluent resources. A combined set of Fluent
resources for a given locale (e.g. `en-US`) is known as Fluent bundle.

```fluent
typeck_address_of_temporary_taken = cannot take address of a temporary
```

In the above example, `typeck_address_of_temporary_taken` is the identifier for
a Fluent message and corresponds to the diagnostic message in English. Other
Fluent resources can be written which would correspond to a message in another
language. Each diagnostic therefore has at least one Fluent message.

```fluent
typeck_address_of_temporary_taken = cannot take address of a temporary
    .label = temporary value
```

By convention, diagnostic messages for subdiagnostics are specified as
"attributes" on Fluent messages (additional related messages, denoted by the
`.<attribute-name>` syntax). In the above example, `label` is an attribute of
`typeck_address_of_temporary_taken` which corresponds to the message for the
label added to this diagnostic.

Diagnostic messages often interpolate additional context into the message shown
to the user, such as the name of a type or of a variable. Additional context to
Fluent messages is provided as an "argument" to the diagnostic.

```fluent
typeck_struct_expr_non_exhaustive =
    cannot create non-exhaustive {$what} using struct expression
```

In the above example, the Fluent message refers to an argument named `what`
which is expected to exist (how arguments are provided to diagnostics is
discussed in detail later).

You can consult the [Fluent] documentation for other usage examples of Fluent
and its syntax.

### Guideline for message naming

Usually, fluent uses `-` for separating words inside a message name. However,
`_` is accepted by fluent as well. As `_` fits Rust's use cases better, due to
the identifiers on the Rust side using `_` as well, inside rustc, `-` is not
allowed for separating words, and instead `_` is recommended. The only exception
is for leading `-`s, for message names like `-passes_see_issue`.

### Guidelines for writing translatable messages

For a message to be translatable into different languages, all of the
information required by any language must be provided to the diagnostic as an
argument (not just the information required in the English message).

As the compiler team gain more experience writing diagnostics that have all of
the information necessary to be translated into different languages, this page
will be updated with more guidance. For now, the [Fluent] documentation has
excellent examples of translating messages into different locales and the
information that needs to be provided by the code to do so.

### Compile-time validation and typed identifiers

rustc's `#[derive(Diagnostic)]` macro performs compile-time validation of Fluent
messages. Compile-time validation of Fluent resources will emit any parsing errors
from Fluent resources while building the compiler, preventing invalid Fluent
resources from causing panics in the compiler. Compile-time validation also
emits an error if multiple Fluent messages have the same identifier.

## Internals

Various parts of rustc's diagnostic internals are modified in order to support
translation.

### Messages

All of rustc's traditional diagnostic APIs (e.g. `struct_span_err` or `note`)
take any message that can be converted into a `DiagMessage`.

[`rustc_error_messages::DiagMessage`] can represent legacy non-translatable
diagnostic messages and translatable messages. Non-translatable messages are
just `String`s. Translatable messages are just a `&'static str` with the
identifier of the Fluent message (sometimes with an additional `&'static str`
with an attribute).

`DiagMessage` never needs to be interacted with directly:
`DiagMessage` constants are created for each diagnostic message in a
Fluent resource (described in more detail below), or `DiagMessage`s will
either be created in the macro-generated code of a diagnostic derive.

`DiagMessage`  implements `Into` for any
type that can be converted into a string, and converts these into
non-translatable diagnostics - this keeps all existing diagnostic calls
working.

### Arguments

Additional context for Fluent messages which are interpolated into message
contents needs to be provided to translatable diagnostics.

Diagnostics have a `set_arg` function that can be used to provide this
additional context to a diagnostic.

Arguments have both a name (e.g. "what" in the earlier example) and a value.
Argument values are represented using the `DiagArgValue` type, which is
just a string or a number. rustc types can implement `IntoDiagArg` with
conversion into a string or a number, and common types like `Ty<'tcx>` already
have such implementations.

`set_arg` calls are handled transparently by diagnostic derives but need to be
added manually when using diagnostic builder APIs.

[Fluent]: https://projectfluent.org
[`compiler/rustc_borrowck/messages.ftl`]: https://github.com/rust-lang/rust/blob/HEAD/compiler/rustc_borrowck/messages.ftl
[`compiler/rustc_parse/messages.ftl`]: https://github.com/rust-lang/rust/blob/HEAD/compiler/rustc_parse/messages.ftl
[`rustc_error_messages::DiagMessage`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_error_messages/enum.DiagMessage.html
