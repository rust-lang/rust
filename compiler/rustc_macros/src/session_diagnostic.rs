#![deny(unused_must_use)]
use proc_macro::Diagnostic;
use quote::{format_ident, quote};
use syn::spanned::Spanned;

use std::collections::{BTreeSet, HashMap};

/// Implements #[derive(SessionDiagnostic)], which allows for errors to be specified as a struct, independent
/// from the actual diagnostics emitting code.
/// ```ignore (pseudo-rust)
/// # extern crate rustc_errors;
/// # use rustc_errors::Applicability;
/// # extern crate rustc_span;
/// # use rustc_span::{symbol::Ident, Span};
/// # extern crate rust_middle;
/// # use rustc_middle::ty::Ty;
/// #[derive(SessionDiagnostic)]
/// #[code = "E0505"]
/// #[error = "cannot move out of {name} because it is borrowed"]
/// pub struct MoveOutOfBorrowError<'tcx> {
///     pub name: Ident,
///     pub ty: Ty<'tcx>,
///     #[label = "cannot move out of borrow"]
///     pub span: Span,
///     #[label = "`{ty}` first borrowed here"]
///     pub other_span: Span,
///     #[suggestion(message = "consider cloning here", code = "{name}.clone()")]
///     pub opt_sugg: Option<(Span, Applicability)>
/// }
/// ```
/// Then, later, to emit the error:
///
/// ```ignore (pseudo-rust)
/// sess.emit_err(MoveOutOfBorrowError {
///     expected,
///     actual,
///     span,
///     other_span,
///     opt_sugg: Some(suggestion, Applicability::MachineApplicable),
/// });
/// ```
pub fn session_diagnostic_derive(s: synstructure::Structure<'_>) -> proc_macro2::TokenStream {
    // Names for the diagnostic we build and the session we build it from.
    let diag = format_ident!("diag");
    let sess = format_ident!("sess");

    SessionDiagnosticDerive::new(diag, sess, s).into_tokens()
}

// Checks whether the type name of `ty` matches `name`.
//
// Given some struct at a::b::c::Foo, this will return true for c::Foo, b::c::Foo, or
// a::b::c::Foo. This reasonably allows qualified names to be used in the macro.
fn type_matches_path(ty: &syn::Type, name: &[&str]) -> bool {
    if let syn::Type::Path(ty) = ty {
        ty.path
            .segments
            .iter()
            .map(|s| s.ident.to_string())
            .rev()
            .zip(name.iter().rev())
            .all(|(x, y)| &x.as_str() == y)
    } else {
        false
    }
}

/// The central struct for constructing the as_error method from an annotated struct.
struct SessionDiagnosticDerive<'a> {
    structure: synstructure::Structure<'a>,
    builder: SessionDiagnosticDeriveBuilder<'a>,
}

impl std::convert::From<syn::Error> for SessionDiagnosticDeriveError {
    fn from(e: syn::Error) -> Self {
        SessionDiagnosticDeriveError::SynError(e)
    }
}

/// Equivalent to rustc:errors::diagnostic::DiagnosticId, except stores the quoted expression to
/// initialise the code with.
enum DiagnosticId {
    Error(proc_macro2::TokenStream),
    Lint(proc_macro2::TokenStream),
}

#[derive(Debug)]
enum SessionDiagnosticDeriveError {
    SynError(syn::Error),
    ErrorHandled,
}

impl SessionDiagnosticDeriveError {
    fn to_compile_error(self) -> proc_macro2::TokenStream {
        match self {
            SessionDiagnosticDeriveError::SynError(e) => e.to_compile_error(),
            SessionDiagnosticDeriveError::ErrorHandled => {
                // Return ! to avoid having to create a blank DiagnosticBuilder to return when an
                // error has already been emitted to the compiler.
                quote! {
                    unreachable!()
                }
            }
        }
    }
}

fn span_err(span: impl proc_macro::MultiSpan, msg: &str) -> proc_macro::Diagnostic {
    Diagnostic::spanned(span, proc_macro::Level::Error, msg)
}

/// For methods that return a Result<_, SessionDiagnosticDeriveError>: emit a diagnostic on
/// span $span with msg $msg (and, optionally, perform additional decoration using the FnOnce
/// passed in `diag`). Then, return Err(ErrorHandled).
macro_rules! throw_span_err {
    ($span:expr, $msg:expr) => {{ throw_span_err!($span, $msg, |diag| diag) }};
    ($span:expr, $msg:expr, $f:expr) => {{
        return Err(_throw_span_err($span, $msg, $f));
    }};
}

/// When possible, prefer using throw_span_err! over using this function directly. This only exists
/// as a function to constrain `f` to an impl FnOnce.
fn _throw_span_err(
    span: impl proc_macro::MultiSpan,
    msg: &str,
    f: impl FnOnce(proc_macro::Diagnostic) -> proc_macro::Diagnostic,
) -> SessionDiagnosticDeriveError {
    let diag = span_err(span, msg);
    f(diag).emit();
    SessionDiagnosticDeriveError::ErrorHandled
}

impl<'a> SessionDiagnosticDerive<'a> {
    fn new(diag: syn::Ident, sess: syn::Ident, structure: synstructure::Structure<'a>) -> Self {
        // Build the mapping of field names to fields. This allows attributes to peek values from
        // other fields.
        let mut fields_map = HashMap::new();

        // Convenience bindings.
        let ast = structure.ast();

        if let syn::Data::Struct(syn::DataStruct { fields, .. }) = &ast.data {
            for field in fields.iter() {
                if let Some(ident) = &field.ident {
                    fields_map.insert(ident.to_string(), field);
                }
            }
        }

        Self {
            builder: SessionDiagnosticDeriveBuilder { diag, sess, fields: fields_map, kind: None },
            structure,
        }
    }
    fn into_tokens(self) -> proc_macro2::TokenStream {
        let SessionDiagnosticDerive { structure, mut builder } = self;

        let ast = structure.ast();
        let attrs = &ast.attrs;

        let implementation = {
            if let syn::Data::Struct(..) = ast.data {
                let preamble = {
                    let preamble = attrs.iter().map(|attr| {
                        builder
                            .generate_structure_code(attr)
                            .unwrap_or_else(|v| v.to_compile_error())
                    });
                    quote! {
                        #(#preamble)*;
                    }
                };

                let body = structure.each(|field_binding| {
                    let field = field_binding.ast();
                    let result = field.attrs.iter().map(|attr| {
                        builder
                            .generate_field_code(
                                attr,
                                FieldInfo {
                                    vis: &field.vis,
                                    binding: field_binding,
                                    ty: &field.ty,
                                    span: &field.span(),
                                },
                            )
                            .unwrap_or_else(|v| v.to_compile_error())
                    });
                    return quote! {
                        #(#result);*
                    };
                });
                // Finally, putting it altogether.
                match builder.kind {
                    None => {
                        span_err(ast.span().unwrap(), "`code` not specified")
                        .help("use the [code = \"...\"] attribute to set this diagnostic's error code ")
                        .emit();
                        SessionDiagnosticDeriveError::ErrorHandled.to_compile_error()
                    }
                    Some((kind, _)) => match kind {
                        DiagnosticId::Lint(_lint) => todo!(),
                        DiagnosticId::Error(code) => {
                            let (diag, sess) = (&builder.diag, &builder.sess);
                            quote! {
                                let mut #diag = #sess.struct_err_with_code("", rustc_errors::DiagnosticId::Error(#code));
                                #preamble
                                match self {
                                    #body
                                }
                                #diag
                            }
                        }
                    },
                }
            } else {
                span_err(
                    ast.span().unwrap(),
                    "`#[derive(SessionDiagnostic)]` can only be used on structs",
                )
                .emit();
                SessionDiagnosticDeriveError::ErrorHandled.to_compile_error()
            }
        };

        let sess = &builder.sess;
        structure.gen_impl(quote! {
            gen impl<'__session_diagnostic_sess> rustc_session::SessionDiagnostic<'__session_diagnostic_sess>
                    for @Self
            {
                fn into_diagnostic(
                    self,
                    #sess: &'__session_diagnostic_sess rustc_session::Session
                ) -> rustc_errors::DiagnosticBuilder<'__session_diagnostic_sess> {
                    #implementation
                }
            }
        })
    }
}

/// Field information passed to the builder. Deliberately omits attrs to discourage the generate_*
/// methods from walking the attributes themselves.
struct FieldInfo<'a> {
    vis: &'a syn::Visibility,
    binding: &'a synstructure::BindingInfo<'a>,
    ty: &'a syn::Type,
    span: &'a proc_macro2::Span,
}

/// Tracks persistent information required for building up the individual calls to diagnostic
/// methods for the final generated method. This is a separate struct to SessionDerive only to be
/// able to destructure and split self.builder and the self.structure up to avoid a double mut
/// borrow later on.
struct SessionDiagnosticDeriveBuilder<'a> {
    /// Name of the session parameter that's passed in to the as_error method.
    sess: syn::Ident,

    /// Store a map of field name to its corresponding field. This is built on construction of the
    /// derive builder.
    fields: HashMap<String, &'a syn::Field>,

    /// The identifier to use for the generated DiagnosticBuilder instance.
    diag: syn::Ident,

    /// Whether this is a lint or an error. This dictates how the diag will be initialised. Span
    /// stores at what Span the kind was first set at (for error reporting purposes, if the kind
    /// was multiply specified).
    kind: Option<(DiagnosticId, proc_macro2::Span)>,
}

impl<'a> SessionDiagnosticDeriveBuilder<'a> {
    fn generate_structure_code(
        &mut self,
        attr: &syn::Attribute,
    ) -> Result<proc_macro2::TokenStream, SessionDiagnosticDeriveError> {
        Ok(match attr.parse_meta()? {
            syn::Meta::NameValue(syn::MetaNameValue { lit: syn::Lit::Str(s), .. }) => {
                let formatted_str = self.build_format(&s.value(), attr.span());
                let name = attr.path.segments.last().unwrap().ident.to_string();
                let name = name.as_str();
                match name {
                    "message" => {
                        let diag = &self.diag;
                        quote! {
                            #diag.set_primary_message(#formatted_str);
                        }
                    }
                    attr @ "error" | attr @ "lint" => {
                        self.set_kind_once(
                            if attr == "error" {
                                DiagnosticId::Error(formatted_str)
                            } else if attr == "lint" {
                                DiagnosticId::Lint(formatted_str)
                            } else {
                                unreachable!()
                            },
                            s.span(),
                        )?;
                        // This attribute is only allowed to be applied once, and the attribute
                        // will be set in the initialisation code.
                        quote! {}
                    }
                    other => throw_span_err!(
                        attr.span().unwrap(),
                        &format!(
                            "`#[{} = ...]` is not a valid SessionDiagnostic struct attribute",
                            other
                        )
                    ),
                }
            }
            _ => todo!("unhandled meta kind"),
        })
    }

    #[must_use]
    fn set_kind_once(
        &mut self,
        kind: DiagnosticId,
        span: proc_macro2::Span,
    ) -> Result<(), SessionDiagnosticDeriveError> {
        if self.kind.is_none() {
            self.kind = Some((kind, span));
            Ok(())
        } else {
            let kind_str = |kind: &DiagnosticId| match kind {
                DiagnosticId::Lint(..) => "lint",
                DiagnosticId::Error(..) => "error",
            };

            let existing_kind = kind_str(&self.kind.as_ref().unwrap().0);
            let this_kind = kind_str(&kind);

            let msg = if this_kind == existing_kind {
                format!("`{}` specified multiple times", existing_kind)
            } else {
                format!("`{}` specified when `{}` was already specified", this_kind, existing_kind)
            };
            throw_span_err!(span.unwrap(), &msg);
        }
    }

    fn generate_field_code(
        &mut self,
        attr: &syn::Attribute,
        info: FieldInfo<'_>,
    ) -> Result<proc_macro2::TokenStream, SessionDiagnosticDeriveError> {
        let field_binding = &info.binding.binding;

        let option_ty = option_inner_ty(info.ty);

        let generated_code = self.generate_non_option_field_code(
            attr,
            FieldInfo {
                vis: info.vis,
                binding: info.binding,
                ty: option_ty.unwrap_or(info.ty),
                span: info.span,
            },
        )?;
        Ok(if option_ty.is_none() {
            quote! { #generated_code }
        } else {
            quote! {
                if let Some(#field_binding) = #field_binding {
                    #generated_code
                }
            }
        })
    }

    fn generate_non_option_field_code(
        &mut self,
        attr: &syn::Attribute,
        info: FieldInfo<'_>,
    ) -> Result<proc_macro2::TokenStream, SessionDiagnosticDeriveError> {
        let diag = &self.diag;
        let field_binding = &info.binding.binding;
        let name = attr.path.segments.last().unwrap().ident.to_string();
        let name = name.as_str();
        // At this point, we need to dispatch based on the attribute key + the
        // type.
        let meta = attr.parse_meta()?;
        Ok(match meta {
            syn::Meta::NameValue(syn::MetaNameValue { lit: syn::Lit::Str(s), .. }) => {
                let formatted_str = self.build_format(&s.value(), attr.span());
                match name {
                    "message" => {
                        if type_matches_path(info.ty, &["rustc_span", "Span"]) {
                            quote! {
                                #diag.set_span(*#field_binding);
                                #diag.set_primary_message(#formatted_str);
                            }
                        } else {
                            throw_span_err!(
                                attr.span().unwrap(),
                                "the `#[message = \"...\"]` attribute can only be applied to fields of type Span"
                            );
                        }
                    }
                    "label" => {
                        if type_matches_path(info.ty, &["rustc_span", "Span"]) {
                            quote! {
                                #diag.span_label(*#field_binding, #formatted_str);
                            }
                        } else {
                            throw_span_err!(
                                attr.span().unwrap(),
                                "The `#[label = ...]` attribute can only be applied to fields of type Span"
                            );
                        }
                    }
                    other => throw_span_err!(
                        attr.span().unwrap(),
                        &format!(
                            "`#[{} = ...]` is not a valid SessionDiagnostic field attribute",
                            other
                        )
                    ),
                }
            }
            syn::Meta::List(list) => {
                match list.path.segments.iter().last().unwrap().ident.to_string().as_str() {
                    suggestion_kind @ "suggestion"
                    | suggestion_kind @ "suggestion_short"
                    | suggestion_kind @ "suggestion_hidden"
                    | suggestion_kind @ "suggestion_verbose" => {
                        // For suggest, we need to ensure we are running on a (Span,
                        // Applicability) pair.
                        let (span, applicability) = (|| match &info.ty {
                            ty @ syn::Type::Path(..)
                                if type_matches_path(ty, &["rustc_span", "Span"]) =>
                            {
                                let binding = &info.binding.binding;
                                Ok((
                                    quote!(*#binding),
                                    quote!(rustc_errors::Applicability::Unspecified),
                                ))
                            }
                            syn::Type::Tuple(tup) => {
                                let mut span_idx = None;
                                let mut applicability_idx = None;
                                for (idx, elem) in tup.elems.iter().enumerate() {
                                    if type_matches_path(elem, &["rustc_span", "Span"]) {
                                        if span_idx.is_none() {
                                            span_idx = Some(syn::Index::from(idx));
                                        } else {
                                            throw_span_err!(
                                                info.span.unwrap(),
                                                "type of field annotated with `#[suggestion(...)]` contains more than one Span"
                                            );
                                        }
                                    } else if type_matches_path(
                                        elem,
                                        &["rustc_errors", "Applicability"],
                                    ) {
                                        if applicability_idx.is_none() {
                                            applicability_idx = Some(syn::Index::from(idx));
                                        } else {
                                            throw_span_err!(
                                                info.span.unwrap(),
                                                "type of field annotated with `#[suggestion(...)]` contains more than one Applicability"
                                            );
                                        }
                                    }
                                }
                                if let Some(span_idx) = span_idx {
                                    let binding = &info.binding.binding;
                                    let span = quote!(#binding.#span_idx);
                                    let applicability = applicability_idx
                                        .map(
                                            |applicability_idx| quote!(#binding.#applicability_idx),
                                        )
                                        .unwrap_or_else(|| {
                                            quote!(rustc_errors::Applicability::Unspecified)
                                        });
                                    return Ok((span, applicability));
                                }
                                throw_span_err!(
                                    info.span.unwrap(),
                                    "wrong types for suggestion",
                                    |diag| {
                                        diag.help("#[suggestion(...)] on a tuple field must be applied to fields of type (Span, Applicability)")
                                    }
                                );
                            }
                            _ => throw_span_err!(
                                info.span.unwrap(),
                                "wrong field type for suggestion",
                                |diag| {
                                    diag.help("#[suggestion(...)] should be applied to fields of type Span or (Span, Applicability)")
                                }
                            ),
                        })()?;
                        // Now read the key-value pairs.
                        let mut msg = None;
                        let mut code = None;

                        for arg in list.nested.iter() {
                            if let syn::NestedMeta::Meta(syn::Meta::NameValue(arg_name_value)) = arg
                            {
                                if let syn::MetaNameValue { lit: syn::Lit::Str(s), .. } =
                                    arg_name_value
                                {
                                    let name = arg_name_value
                                        .path
                                        .segments
                                        .last()
                                        .unwrap()
                                        .ident
                                        .to_string();
                                    let name = name.as_str();
                                    let formatted_str = self.build_format(&s.value(), arg.span());
                                    match name {
                                        "message" => {
                                            msg = Some(formatted_str);
                                        }
                                        "code" => {
                                            code = Some(formatted_str);
                                        }
                                        other => throw_span_err!(
                                            arg.span().unwrap(),
                                            &format!(
                                                "`{}` is not a valid key for `#[suggestion(...)]`",
                                                other
                                            )
                                        ),
                                    }
                                }
                            }
                        }
                        let msg = if let Some(msg) = msg {
                            quote!(#msg.as_str())
                        } else {
                            throw_span_err!(
                                list.span().unwrap(),
                                "missing suggestion message",
                                |diag| {
                                    diag.help("provide a suggestion message using #[suggestion(message = \"...\")]")
                                }
                            );
                        };
                        let code = code.unwrap_or_else(|| quote! { String::new() });
                        // Now build it out:
                        let suggestion_method = format_ident!("span_{}", suggestion_kind);
                        quote! {
                            #diag.#suggestion_method(#span, #msg, #code, #applicability);
                        }
                    }
                    other => throw_span_err!(
                        list.span().unwrap(),
                        &format!("invalid annotation list `#[{}(...)]`", other)
                    ),
                }
            }
            _ => panic!("unhandled meta kind"),
        })
    }

    /// In the strings in the attributes supplied to this macro, we want callers to be able to
    /// reference fields in the format string. Take this, for example:
    /// ```ignore (not-usage-example)
    /// struct Point {
    ///     #[error = "Expected a point greater than ({x}, {y})"]
    ///     x: i32,
    ///     y: i32,
    /// }
    /// ```
    /// We want to automatically pick up that {x} refers `self.x` and {y} refers to `self.y`, then
    /// generate this call to format!:
    /// ```ignore (not-usage-example)
    /// format!("Expected a point greater than ({x}, {y})", x = self.x, y = self.y)
    /// ```
    /// This function builds the entire call to format!.
    fn build_format(&self, input: &str, span: proc_macro2::Span) -> proc_macro2::TokenStream {
        // This set is used later to generate the final format string. To keep builds reproducible,
        // the iteration order needs to be deterministic, hence why we use a BTreeSet here instead
        // of a HashSet.
        let mut referenced_fields: BTreeSet<String> = BTreeSet::new();

        // At this point, we can start parsing the format string.
        let mut it = input.chars().peekable();
        // Once the start of a format string has been found, process the format string and spit out
        // the referenced fields. Leaves `it` sitting on the closing brace of the format string, so the
        // next call to `it.next()` retrieves the next character.
        while let Some(c) = it.next() {
            if c == '{' && *it.peek().unwrap_or(&'\0') != '{' {
                #[must_use]
                let mut eat_argument = || -> Option<String> {
                    let mut result = String::new();
                    // Format specifiers look like
                    // format   := '{' [ argument ] [ ':' format_spec ] '}' .
                    // Therefore, we only need to eat until ':' or '}' to find the argument.
                    while let Some(c) = it.next() {
                        result.push(c);
                        let next = *it.peek().unwrap_or(&'\0');
                        if next == '}' {
                            break;
                        } else if next == ':' {
                            // Eat the ':' character.
                            assert_eq!(it.next().unwrap(), ':');
                            break;
                        }
                    }
                    // Eat until (and including) the matching '}'
                    while it.next()? != '}' {
                        continue;
                    }
                    Some(result)
                };

                if let Some(referenced_field) = eat_argument() {
                    referenced_fields.insert(referenced_field);
                }
            }
        }
        // At this point, `referenced_fields` contains a set of the unique fields that were
        // referenced in the format string. Generate the corresponding "x = self.x" format
        // string parameters:
        let args = referenced_fields.into_iter().map(|field: String| {
            let field_ident = format_ident!("{}", field);
            let value = if self.fields.contains_key(&field) {
                quote! {
                    &self.#field_ident
                }
            } else {
                // This field doesn't exist. Emit a diagnostic.
                Diagnostic::spanned(
                    span.unwrap(),
                    proc_macro::Level::Error,
                    format!("`{}` doesn't refer to a field on this type", field),
                )
                .emit();
                quote! {
                    "{#field}"
                }
            };
            quote! {
                #field_ident = #value
            }
        });
        quote! {
            format!(#input #(,#args)*)
        }
    }
}

/// If `ty` is an Option, returns Some(inner type). Else, returns None.
fn option_inner_ty(ty: &syn::Type) -> Option<&syn::Type> {
    if type_matches_path(ty, &["std", "option", "Option"]) {
        if let syn::Type::Path(ty_path) = ty {
            let path = &ty_path.path;
            let ty = path.segments.iter().last().unwrap();
            if let syn::PathArguments::AngleBracketed(bracketed) = &ty.arguments {
                if bracketed.args.len() == 1 {
                    if let syn::GenericArgument::Type(ty) = &bracketed.args[0] {
                        return Some(ty);
                    }
                }
            }
        }
    }
    None
}
