# Creating translatable errors using `SessionDiagnostic`
The `SessionDiagnostic` derive macro is the recommended way to create
diagnostics. Diagnostics created with the derive macro can be translated into
different languages and each have a slug that uniquely identifies the
diagnostic.

Instead of using the `DiagnosticBuilder` API to create and emit diagnostics,
the `SessionDiagnostic` derive macro is applied to structs.

The [definition][defn] of the "field already declared" diagnostic is shown
below.

```rust,ignore
#[derive(SessionDiagnostic)]
#[error(code = "E0124", slug = "typeck-field-already-declared")]
pub struct FieldAlreadyDeclared {
    pub field_name: Ident,
    #[primary_span]
    #[label]
    pub span: Span,
    #[label = "previous-decl-label"]
    pub prev_span: Span,
}
```

Every `SessionDiagnostic` has to have one attribute applied to the struct
itself: either `#[error(..)]` for defining errors, or `#[warning(..)]` for
defining warnings.

If an error has an error code (e.g. "E0624"), then that can be specified using
the `code` sub-attribute. Specifying a `code` isn't mandatory, but if you are
porting a diagnostic that uses `DiagnosticBuilder` to use `SessionDiagnostic`
then you should keep the code if there was one.

Both `#[error(..)]` and `#[warning(..)]` must set a value for the `slug`
sub-attribute. `slug` uniquely identifies the diagnostic and is also how the
compiler knows what error message to emit (in the default locale of the
compiler, or in the locale requested by the user).

rustc uses [Fluent](https://projectfluent.org) to handle the intricacies of
translation. Each diagnostic's `slug` is actually an identifier for a *Fluent
message*. Let's take a look at what the Fluent message for the "field already
declared" diagnostic looks like:

```fluent
typeck-field-already-declared =
    field `{$field_name}` is already declared
    .label = field already declared
    .previous-decl-label = `{$field_name}` first declared here
```

`typeck-field-already-declared` is the `slug` from our example and is followed
by the diagnostic message.

Fluent is built around the idea of "asymmetric localization", which aims to
decouple the expressiveness of translations from the grammar of the source
language (English in rustc's case). Prior to translation, rustc's diagnostics
relied heavily on interpolation to build the messages shown to the users.
Interpolated strings are hard to translate because writing a natural-sounding
translation might require more, less, or just different interpolation than the
English string, all of which would require changes to the compiler's source
code to support.

As the compiler team gain more experience creating `SessionDiagnostic` structs
that have all of the information necessary to be translated into different
languages, this page will be updated with more guidance. For now, the [Project
Fluent](https://projectfluent.org) documentation has excellent examples of
translating messages into different locales and the information that needs to
be provided by the code to do so.

When adding or changing a diagnostic, you don't need to worry about the
translations, only updating the original English message. All of rustc's
English Fluent messages can be found in
`/compiler/rustc_error_messages/locales/en-US/diagnostics.ftl`.

Every field of the `SessionDiagnostic` which does not have an annotation is
available in Fluent messages as a variable, like `field_name` in the example
above.

Using the `#[primary_span]` attribute on a field (that has type `Span`)
indicates the primary span of the diagnostic which will have the main message
of the diagnostic.

Diagnostics are more than just their primary message, they often include
labels, notes, help messages and suggestions, all of which can also be
specified on a `SessionDiagnostic`.

`#[label]`, `#[help]` and `#[note]` can all be applied to fields which have the
type `Span`. Applying any of these attributes will create the corresponding
sub-diagnostic with that `Span`. These attributes will look for their
diagnostic message in a Fluent attribute attached to the primary Fluent
message. In our example, `#[label]` will look for
`typeck-field-already-declared.label` (which has the message "field already
declared"). If there is more than one sub-diagnostic of the same type, then
these attributes can also take a value that is the attribute name to look for
(e.g. `previous-decl-label` in our example).

`#[help]` and `#[note]` can also be applied to the struct itself, in which case
they work exactly like when applied to fields except the sub-diagnostic won't
have a `Span`.

Any attribute can also be applied to an `Option<Span>` and will only emit a
sub-diagnostic if the option is `Some(..)`.

Suggestions can be emitted using one of four field attributes:

- `#[suggestion(message = "...", code = "...")]`
- `#[suggestion_hidden(message = "...", code = "...")]`
- `#[suggestion_short(message = "...", code = "...")]`
- `#[suggestion_verbose(message = "...", code = "...")]`

Suggestions must be applied on either a `Span` field or a
`(Span, MachineApplicability)` field. Similarly to other field attributes,
`message` specifies the Fluent attribute with the message and defaults to
`.suggestion`. `code` specifies the code that should be suggested as a
replacement and is a format string (e.g. `{field_name}` would be replaced by
the value of the `field_name` field of the struct), not a Fluent identifier.

In the end, the `SessionDiagnostic` derive will generate an implementation of
`SessionDiagnostic` that looks like the following:

```rust,ignore
impl SessionDiagnostic for FieldAlreadyDeclared {
    fn into_diagnostic(self, sess: &'_ rustc_session::Session) -> DiagnosticBuilder<'_> {
        let mut diag = sess.struct_err_with_code(
            rustc_errors::DiagnosticMessage::fluent("typeck-field-already-declared"),
            rustc_errors::DiagnosticId::Error("E0124")
        );
        diag.set_span(self.span);
        diag.span_label(
            self.span,
            rustc_errors::DiagnosticMessage::fluent_attr("typeck-field-already-declared", "label")
        );
        diag.span_label(
            self.prev_span,
            rustc_errors::DiagnosticMessage::fluent_attr("typeck-field-already-declared", "previous-decl-label")
        );
        diag
    }
}
```

Now that we've defined our diagnostic, how do we [use it][use]? It's quite
straightforward, just create an instance of the struct and pass it to
`emit_err` (or `emit_warning`):

```rust,ignore
tcx.sess.emit_err(FieldAlreadyDeclared {
    field_name: f.ident,
    span: f.span,
    prev_span,
});
```

## Reference
`#[derive(SessionDiagnostic)]` supports the following attributes:

- `#[error(code = "...", slug = "...")]` or `#[warning(code = "...", slug = "...")]`
  - _Applied to struct._
  - _Mandatory_
  - Defines the struct to be representing an error or a warning.
  - `code = "..."`
    - _Optional_
    - Specifies the error code.
  - `slug = "..."`
    - _Mandatory_
    - Uniquely identifies the diagnostic and corresponds to its Fluent message,
      mandatory.
- `#[note]` or `#[note = "..."]`
  - _Applied to struct or `Span` fields._
  - _Optional_
  - Adds a note sub-diagnostic.
  - Value is the Fluent attribute (relative to the Fluent message specified by
    `slug`) for the note's message
    - Defaults to `note`.
  - If applied to a `Span` field, creates a spanned note.
- `#[help]` or `#[help = "..."]`
  - _Applied to struct or `Span` fields._
  - _Optional_
  - Adds a help sub-diagnostic.
  - Value is the Fluent attribute (relative to the Fluent message specified by
    `slug`) for the help's message
    - Defaults to `help`.
  - If applied to a `Span` field, creates a spanned help.
- `#[label]` or `#[label = "..."]`
  - _Applied to `Span` fields._
  - _Optional_
  - Adds a label sub-diagnostic.
  - Value is the Fluent attribute (relative to the Fluent message specified by
    `slug`) for the label's message
    - Defaults to `label`.
- `#[suggestion{,_hidden,_short,_verbose}(message = "...", code = "...")]`
  - _Applied to `(Span, MachineApplicability)` or `Span` fields._
  - _Optional_
  - Adds a suggestion sub-diagnostic.
  - `message = "..."`
    - _Mandatory_
    - Value is the Fluent attribute (relative to the Fluent message specified
      by `slug`) for the suggestion's message
    - Defaults to `suggestion`.
  - `code = "..."`
    - _Optional_
    - Value is a format string indicating the code to be suggested as a
      replacement.

[defn]: https://github.com/rust-lang/rust/blob/bbe9d27b8ff36da56638aa43d6d0cdfdf89a4e57/compiler/rustc_typeck/src/errors.rs#L65-L74
[use]: https://github.com/rust-lang/rust/blob/eb82facb1626166188d49599a3313fc95201f556/compiler/rustc_typeck/src/collect.rs#L981-L985
