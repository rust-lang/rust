# Creating Errors With SessionDiagnostic

The SessionDiagnostic derive macro gives an alternate way to the DiagnosticBuilder API for defining
and emitting errors.  It allows a struct to be annotated with information which allows it to be
transformed and emitted as a Diagnostic.

As an example, we'll take a look at how the "field already declared" diagnostic is actually defined
in the compiler (see the definition
[here](https://github.com/rust-lang/rust/blob/75042566d1c90d912f22e4db43b6d3af98447986/compiler/rustc_typeck/src/errors.rs#L65-L74)
and usage
[here](https://github.com/rust-lang/rust/blob/75042566d1c90d912f22e4db43b6d3af98447986/compiler/rustc_typeck/src/collect.rs#L863-L867)):

```rust,ignore
#[derive(SessionDiagnostic)]
#[error = "E0124"]
pub struct FieldAlreadyDeclared {
    pub field_name: Ident,
    #[message = "field `{field_name}` is already declared"]
    #[label = "field already declared"]
    pub span: Span,
    #[label = "`{field_name}` first declared here"]
    pub prev_span: Span,
}
// ...
tcx.sess.emit_err(FieldAlreadyDeclared {
    field_name: f.ident,
    span: f.span,
    prev_span,
});
```

We see that using `SessionDiagnostic` is relatively straight forward.  The `#[error = "..."]`
attribute is used to supply the error code for the diagnostic.  We then annotate fields in the
struct with various information on how to convert an instance of the struct into a rendered
diagnostic.  The attributes above produce code which is roughly equivalent to the following (in
pseudo-Rust):

```rust,ignore
impl SessionDiagnostic for FieldAlreadyDeclared {
    fn into_diagnostic(self, sess: &'_ rustc_session::Session) -> DiagnosticBuilder<'_> {
        let mut diag = sess.struct_err_with_code("", rustc_errors::DiagnosticId::Error("E0124"));
        diag.set_span(self.span);
        diag.set_primary_message(format!("field `{field_name}` is already declared", field_name = self.field_name));
        diag.span_label(self.span, "field already declared");
        diag.span_label(self.prev_span, format!("`{field_name}` first declared here", field_name = self.field_name));
        diag
    }
}
```

The generated code draws attention to a number of features.  First, we see that within the strings
passed to each attribute, field names can be referenced without needing to be passed
explicitly into the format string -- in this example here, `#[message = "field {field_name} is
already declared"]` produces a call to `format!` with the appropriate arguments to format
`self.field_name` into the string.  This applies to strings passed to all attributes.

We also see that labelling `Span` fields in the struct produces calls which pass that `Span` to the
produced diagnostic.  In the example above, we see that putting the `#[message = "..."]` attribute
on a `Span` leads to the primary span of the diagnostic being set to that `Span`, while applying the
`#[label = "..."]` attribute on a Span will simply set the span for that label.
Each attribute has different requirements for what they can be applied on, differing on position
(on the struct, or on a specific field), type (if it's applied on a field), and whether or not the
attribute is optional.

## Attributes Listing

Below is a listing of all the currently-available attributes that `#[derive(SessionDiagnostic)]`
understands:

Attribute                                               | Applied to                                        | Mandatory | Behaviour
:--------------                                         | :--------------------                             |:--------- | :---------
`#[code = "..."]`                                       | Struct                                            | Yes       | Sets the Diagnostic's error code
`#[message = "..."]`                                    | Struct / `Span` fields                            | Yes       | Sets the Diagnostic's primary message. If on `Span` field, also sets the Diagnostic's span.
`#[label = "..."]`                                      | `Span` fields                                     | No        | Equivalent to calling `span_label` with that Span and message.
`#[suggestion(message = "..." , code = "..."]`          | `(Span, MachineApplicability)` or `Span` fields   | No        | Equivalent to calling `span_suggestion`. Note `code` is optional.
`#[suggestion_short(message = "..." , code = "..."]`    | `(Span, MachineApplicability)` or `Span` fields   | No        | Equivalent to calling `span_suggestion_short`. Note `code` is optional.
`#[suggestion_hidden(message = "..." , code = "..."]`   | `(Span, MachineApplicability)` or `Span` fields   | No        | Equivalent to calling `span_suggestion_hidden`. Note `code` is optional.
`#[suggestion_verbose(message = "..." , code = "..."]`  | `(Span, MachineApplicability)` or `Span` fields   | No        | Equivalent to calling `span_suggestion_verbose`. Note `code` is optional.


## Optional Diagnostic Attributes

There may be some cases where you want one of the decoration attributes to be applied optionally;
for example, if a suggestion can only be generated sometimes. In this case, simply wrap the field's
type in an `Option`.  At runtime, if the field is set to `None`, the attribute for that field won't
be used in creating the diagnostic. For example:

```rust,ignored
#[derive(SessionDiagnostic)]
#[code = "E0123"]
struct SomeKindOfError {
    ...
    #[suggestion(message = "informative error message")]
    opt_sugg: Option<(Span, Applicability)>
    ...
}
```
