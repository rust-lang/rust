# Diagnostic and subdiagnostic structs
rustc has three diagnostic traits that can be used to create diagnostics:
`Diagnostic`, `LintDiagnostic`, and `Subdiagnostic`.

For simple diagnostics,
derived impls can be used, e.g. `#[derive(Diagnostic)]`. They are only suitable for simple diagnostics that
don't require much logic in deciding whether or not to add additional subdiagnostics.

In cases where diagnostics require more complex or dynamic behavior, such as conditionally adding subdiagnostics,
customizing the rendering logic, or selecting messages at runtime, you will need to manually implement
the corresponding trait (`Diagnostic`, `LintDiagnostic`, or `Subdiagnostic`).
This approach provides greater flexibility and is recommended for diagnostics that go beyond simple, static structures.

Diagnostic can be translated into different languages.

## `#[derive(Diagnostic)]` and `#[derive(LintDiagnostic)]`

Consider the [definition][defn] of the "field already declared" diagnostic shown below:

```rust,ignore
#[derive(Diagnostic)]
#[diag("field `{$field_name}` is already declared", code = E0124)]
pub struct FieldAlreadyDeclared {
    pub field_name: Ident,
    #[primary_span]
    #[label("field already declared")]
    pub span: Span,
    #[label("`{$field_name}` first declared here")]
    pub prev_span: Span,
}
```

`Diagnostic` can only be derived on structs and enums.
Attributes that are placed on the type for structs are placed on each 
variants for enums (or vice versa).
Each `Diagnostic` has to have one
attribute, `#[diag(...)]`, applied to the struct or each enum variant.

If an error has an error code (e.g. "E0624"), then that can be specified using
the `code` sub-attribute.
Specifying a `code` isn't mandatory, but if you are
porting a diagnostic that uses `Diag` to use `Diagnostic`
then you should keep the code if there was one.

`#[diag(..)]` must provide a message as the first positional argument.
The message is written in English, but might be translated to the locale requested by the user.
See [translation documentation](./translation.md) to learn more about how
translatable error messages are written and how they are generated.

Every field of the `Diagnostic` which does not have an annotation is
available in Fluent messages as a variable, like `field_name` in the example above.
Fields can be annotated `#[skip_arg]` if this is undesired.

Using the `#[primary_span]` attribute on a field (that has type `Span`)
indicates the primary span of the diagnostic which will have the main message of the diagnostic.

Diagnostics are more than just their primary message, they often include
labels, notes, help messages and suggestions, all of which can also be specified on a `Diagnostic`.

`#[label]`, `#[help]`, `#[warning]` and `#[note]` can all be applied to fields which have the
type `Span`.
Applying any of these attributes will create the corresponding subdiagnostic with that `Span`.
These attributes take a diagnostic message as an argument.

Other types have special behavior when used in a `Diagnostic` derive:

- Any attribute applied to an `Option<T>` will only emit a
  subdiagnostic if the option is `Some(..)`.
- Any attribute applied to a `Vec<T>` will be repeated for each element of the vector.

`#[help]`, `#[warning]` and `#[note]` can also be applied to the struct itself, in which case
they work exactly like when applied to fields except the subdiagnostic won't have a `Span`.
These attributes can also be applied to fields of type `()` for
the same effect, which when combined with the `Option` type can be used to
represent optional `#[note]`/`#[help]`/`#[warning]` subdiagnostics.

Suggestions can be emitted using one of four field attributes:

- `#[suggestion("message", code = "...", applicability = "...")]`
- `#[suggestion_hidden("message", code = "...", applicability = "...")]`
- `#[suggestion_short("message", code = "...", applicability = "...")]`
- `#[suggestion_verbose("message", code = "...", applicability = "...")]`

Suggestions must be applied on either a `Span` field or a `(Span,
MachineApplicability)` field.
Similarly to other field attributes, a message needs to be provided which will be shown to the user.
`code` specifies the code that should be suggested as a
replacement and is a format string (e.g. `{field_name}` would be replaced by
the value of the `field_name` field of the struct).
`applicability` can be used to specify the applicability in the attribute, it
cannot be used when the field's type contains an `Applicability`.

In the end, the `Diagnostic` derive will generate an implementation of
`Diagnostic` that looks like the following:

```rust,ignore
impl<'a, G: EmissionGuarantee> Diagnostic<'a> for FieldAlreadyDeclared {
    fn into_diag(self, dcx: &'a DiagCtxt, level: Level) -> Diag<'a, G> {
        let mut diag = Diag::new(dcx, level, "field `{$field_name}` is already declared");
        diag.set_span(self.span);
        diag.span_label(
            self.span,
            "field already declared"
        );
        diag.span_label(
            self.prev_span,
            "`{$field_name}` first declared here"
        );
        diag
    }
}
```

Now that we've defined our diagnostic, how do we [use it][use]?
It's quite straightforward, just create an instance of the struct and pass it to
`emit_err` (or `emit_warning`):

```rust,ignore
tcx.dcx().emit_err(FieldAlreadyDeclared {
    field_name: f.ident,
    span: f.span,
    prev_span,
});
```

### Reference for `#[derive(Diagnostic)]` and `#[derive(LintDiagnostic)]`
`#[derive(Diagnostic)]` and `#[derive(LintDiagnostic)]` support the following attributes:

- `#[diag("message", code = "...")]`
  - _Applied to struct or enum variant._
  - _Mandatory_
  - Defines the text and error code to be associated with the diagnostic.
  - Message (_Mandatory_)
    - The diagnostic message which will be shown to the user.
    - See [translation documentation](./translation.md).
  - `code = "..."` (_Optional_)
    - Specifies the error code.
- `#[note("message")]` (_Optional_)
  - _Applied to struct or struct fields of type `Span`, `Option<()>` or `()`._
  - Adds a note subdiagnostic.
  - Value is the note's message.
  - If applied to a `Span` field, creates a spanned note.
- `#[help("message")]` (_Optional_)
  - _Applied to struct or struct fields of type `Span`, `Option<()>` or `()`._
  - Adds a help subdiagnostic.
  - Value is the help message.
  - If applied to a `Span` field, creates a spanned help.
- `#[label("message")]` (_Optional_)
  - _Applied to `Span` fields._
  - Adds a label subdiagnostic.
  - Value is the label's message.
- `#[warning("message")]` (_Optional_)
  - _Applied to struct or struct fields of type `Span`, `Option<()>` or `()`._
  - Adds a warning subdiagnostic.
  - Value is the warning's message.
- `#[suggestion{,_hidden,_short,_verbose}("message", code = "...", applicability = "...")]`
  (_Optional_)
  - _Applied to `(Span, MachineApplicability)` or `Span` fields._
  - Adds a suggestion subdiagnostic.
  - Message (_Mandatory_)
    - Value is the suggestion message that will be shown to the user.
    - See [translation documentation](./translation.md).
  - `code = "..."`/`code("...", ...)` (_Mandatory_)
    - One or multiple format strings indicating the code to be suggested as a replacement.
      Multiple values signify multiple possible replacements.
  - `applicability = "..."` (_Optional_)
    - String which must be one of `machine-applicable`, `maybe-incorrect`,
      `has-placeholders` or `unspecified`.
- `#[subdiagnostic]`
  - _Applied to a type that implements `Subdiagnostic` (from `#[derive(Subdiagnostic)]`)._
  - Adds the subdiagnostic represented by the subdiagnostic struct.
- `#[primary_span]` (_Optional_)
  - _Applied to `Span` fields on `Subdiagnostic`s.
    Not used for `LintDiagnostic`s._
  - Indicates the primary span of the diagnostic.
- `#[skip_arg]` (_Optional_)
  - _Applied to any field._
  - Prevents the field from being provided as a diagnostic argument.

## `#[derive(Subdiagnostic)]`
It is common in the compiler to write a function that conditionally adds a
specific subdiagnostic to an error if it is applicable.
Oftentimes these subdiagnostics could be represented using a diagnostic struct even if the
overall diagnostic could not.
In this circumstance, the `Subdiagnostic`
derive can be used to represent a partial diagnostic (e.g a note, label, help or
suggestion) as a struct.

Consider the [definition][subdiag_defn] of the "expected return type" label shown below:

```rust
#[derive(Subdiagnostic)]
pub enum ExpectedReturnTypeLabel<'tcx> {
    #[label("expected `()` because of default return type")]
    Unit {
        #[primary_span]
        span: Span,
    },
    #[label("expected `{$expected}` because of return type")]
    Other {
        #[primary_span]
        span: Span,
        expected: Ty<'tcx>,
    },
}
```

Like `Diagnostic`, `Subdiagnostic` can be derived for structs or enums.
Attributes that are placed on the type for structs are placed on each
variants for enums (or vice versa).
Each `Subdiagnostic` should have one attribute applied to the struct or each variant, one of:

- `#[label(..)]` for defining a label
- `#[note(..)]` for defining a note
- `#[help(..)]` for defining a help
- `#[warning(..)]` for defining a warning
- `#[suggestion{,_hidden,_short,_verbose}(..)]` for defining a suggestion

All of the above must provide a diagnostic message as the first positional argument.
See [translation documentation](./translation.md) to learn more about how
translatable error messages are generated.

Using the `#[primary_span]` attribute on a field (with type `Span`) will denote
the primary span of the subdiagnostic.
A primary span is only necessary for a label or suggestion, which can not be spanless.

Every field of the type/variant which does not have an annotation is available
in Fluent messages as a variable.
Fields can be annotated `#[skip_arg]` if this is undesired.

Like `Diagnostic`, `Subdiagnostic` supports `Option<T>` and `Vec<T>` fields.

Suggestions can be emitted using one of four attributes on the type/variant:

- `#[suggestion("...", code = "...", applicability = "...")]`
- `#[suggestion_hidden("...", code = "...", applicability = "...")]`
- `#[suggestion_short("...", code = "...", applicability = "...")]`
- `#[suggestion_verbose("...", code = "...", applicability = "...")]`

Suggestions require `#[primary_span]` be set on a field and can have the following sub-attributes:

- The first positional argument specifies the message which will be shown to the user.
- `code` specifies the code that should be suggested as a replacement and is a
  format string (e.g. `{field_name}` would be replaced by the value of the
  `field_name` field of the struct), not a Fluent identifier.
- `applicability` can be used to specify the applicability in the attribute, it
  cannot be used when the field's type contains an `Applicability`.

Applicabilities can also be specified as a field (of type `Applicability`)
using the `#[applicability]` attribute.

In the end, the `Subdiagnostic` derive will generate an implementation
of `Subdiagnostic` that looks like the following:

```rust
impl<'tcx> Subdiagnostic for ExpectedReturnTypeLabel<'tcx> {
    fn add_to_diag(self, diag: &mut rustc_errors::Diagnostic) {
        use rustc_errors::{Applicability, IntoDiagArg};
        match self {
            ExpectedReturnTypeLabel::Unit { span } => {
                diag.span_label(span, "expected `()` because of default return type")
            }
            ExpectedReturnTypeLabel::Other { span, expected } => {
                diag.set_arg("expected", expected);
                diag.span_label(span, "expected `{$expected}` because of return type")
            }
        }
    }
}
```

Once defined, a subdiagnostic can be used by passing it to the `subdiagnostic`
function ([example][subdiag_use_1] and [example][subdiag_use_2]) on a
diagnostic or by assigning it to a `#[subdiagnostic]`-annotated field of a diagnostic struct.

### Argument sharing and isolation

Subdiagnostics add their own arguments (i.e., certain fields in their structure) to the `Diag` structure before rendering the information.
`Diag` structure also stores the arguments from the main diagnostic, so the subdiagnostic can also use the arguments from the main diagnostic.

However, when a subdiagnostic is added to a main diagnostic by implementing `#[derive(Subdiagnostic)]`,
the following rules, introduced in [rust-lang/rust#142724](https://github.com/rust-lang/rust/pull/142724)
apply to the handling of arguments (i.e., variables used in Fluent messages):

**Argument isolation between sub diagnostics**:
Arguments set by a subdiagnostic are only available during the rendering of that subdiagnostic.
After the subdiagnostic is rendered, all arguments it introduced are restored from the main diagnostic.
This ensures that multiple subdiagnostics do not pollute each other's argument scope.
For example, when using a `Vec<Subdiag>`, it iteratively adds the same argument over and over again.

**Same argument override between sub and main diagnostics**:
If a subdiagnostic sets a argument with the same name as a arguments already in the main diagnostic,
it will report an error at runtime unless both have exactly the same value.
It has two benefits:
- preserves the flexibility that arguments in the main diagnostic are allowed to appear in the attributes of the subdiagnostic.
For example, There is an attribute `#[suggestion("...", code = "{new_vis}")]` in the subdiagnostic, but `new_vis` is the field in the main diagnostic struct.
- prevents accidental overwriting or deletion of arguments required by the main diagnostic or other subdiagnostics.

These rules guarantee that arguments injected by subdiagnostics are strictly scoped to their own rendering.
The main diagnostic's arguments remain unaffected by subdiagnostic logic, even in the presence of name collisions.
Additionally, subdiagnostics can access arguments from the main diagnostic with the same name when needed.

### Reference for `#[derive(Subdiagnostic)]`
`#[derive(Subdiagnostic)]` supports the following attributes:

- `#[label("message")]`, `#[help("message")]`, `#[warning("message")]` or `#[note("message")]`
  - _Applied to struct or enum variant.
    Mutually exclusive with struct/enum variant attributes._
  - _Mandatory_
  - Defines the type to be representing a label, help or note.
  - Message (_Mandatory_)
    - The diagnostic message that will be shown to the user.
    - See [translation documentation](./translation.md).
- `#[suggestion{,_hidden,_short,_verbose}("message", code = "...", applicability = "...")]`
  - _Applied to struct or enum variant.
    Mutually exclusive with struct/enum variant attributes._
  - _Mandatory_
  - Defines the type to be representing a suggestion.
  - Message (_Mandatory_)
    - The diagnostic message that will be shown to the user.
    - See [translation documentation](./translation.md).
  - `code = "..."`/`code("...", ...)` (_Mandatory_)
    - One or multiple format strings indicating the code to be suggested as a replacement.
      Multiple values signify multiple possible replacements.
  - `applicability = "..."` (_Optional_)
    - _Mutually exclusive with `#[applicability]` on a field._
    - Value is the applicability of the suggestion.
    - String which must be one of:
      - `machine-applicable`
      - `maybe-incorrect`
      - `has-placeholders`
      - `unspecified`
- `#[multipart_suggestion{,_hidden,_short,_verbose}("message", applicability = "...")]`
  - _Applied to struct or enum variant.
    Mutually exclusive with struct/enum variant attributes._
  - _Mandatory_
  - Defines the type to be representing a multipart suggestion.
  - Message (_Mandatory_): see `#[suggestion]`
  - `applicability = "..."` (_Optional_): see `#[suggestion]`
- `#[primary_span]` (_Mandatory_ for labels and suggestions; _optional_ otherwise; not applicable
to multipart suggestions)
  - _Applied to `Span` fields._
  - Indicates the primary span of the subdiagnostic.
- `#[suggestion_part(code = "...")]` (_Mandatory_; only applicable to multipart suggestions)
  - _Applied to `Span` fields._
  - Indicates the span to be one part of the multipart suggestion.
  - `code = "..."` (_Mandatory_)
    - Value is a format string indicating the code to be suggested as a replacement.
- `#[applicability]` (_Optional_; only applicable to (simple and multipart) suggestions)
  - _Applied to `Applicability` fields._
  - Indicates the applicability of the suggestion.
- `#[skip_arg]` (_Optional_)
  - _Applied to any field._
  - Prevents the field from being provided as a diagnostic argument.

[defn]: https://github.com/rust-lang/rust/blob/6201eabde85db854c1ebb57624be5ec699246b50/compiler/rustc_hir_analysis/src/errors.rs#L68-L77
[use]: https://github.com/rust-lang/rust/blob/f1112099eba41abadb6f921df7edba70affe92c5/compiler/rustc_hir_analysis/src/collect.rs#L823-L827

[subdiag_defn]: https://github.com/rust-lang/rust/blob/f1112099eba41abadb6f921df7edba70affe92c5/compiler/rustc_hir_analysis/src/errors.rs#L221-L234
[subdiag_use_1]: https://github.com/rust-lang/rust/blob/f1112099eba41abadb6f921df7edba70affe92c5/compiler/rustc_hir_analysis/src/check/fn_ctxt/suggestions.rs#L670-L674
[subdiag_use_2]: https://github.com/rust-lang/rust/blob/f1112099eba41abadb6f921df7edba70affe92c5/compiler/rustc_hir_analysis/src/check/fn_ctxt/suggestions.rs#L704-L707
