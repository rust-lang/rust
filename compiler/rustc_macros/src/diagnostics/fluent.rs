use annotate_snippets::{
    display_list::DisplayList,
    snippet::{Annotation, AnnotationType, Slice, Snippet, SourceAnnotation},
};
use fluent_bundle::{FluentBundle, FluentError, FluentResource};
use fluent_syntax::{
    ast::{
        Attribute, Entry, Expression, Identifier, InlineExpression, Message, Pattern,
        PatternElement,
    },
    parser::ParserError,
};
use proc_macro::{Diagnostic, Level, Span};
use proc_macro2::TokenStream;
use quote::quote;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};
use syn::{parse_macro_input, Ident, LitStr};
use unic_langid::langid;

/// Helper function for returning an absolute path for macro-invocation relative file paths.
///
/// If the input is already absolute, then the input is returned. If the input is not absolute,
/// then it is appended to the directory containing the source file with this macro invocation.
fn invocation_relative_path_to_absolute(span: Span, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        // `/a/b/c/foo/bar.rs` contains the current macro invocation
        let mut source_file_path = span.source_file().path();
        // `/a/b/c/foo/`
        source_file_path.pop();
        // `/a/b/c/foo/../locales/en-US/example.ftl`
        source_file_path.push(path);
        source_file_path
    }
}

/// Tokens to be returned when the macro cannot proceed.
fn failed(crate_name: &Ident) -> proc_macro::TokenStream {
    quote! {
        pub static DEFAULT_LOCALE_RESOURCE: &'static str = "";

        #[allow(non_upper_case_globals)]
        #[doc(hidden)]
        pub(crate) mod fluent_generated {
            pub mod #crate_name {
            }

            pub mod _subdiag {
                pub const help: crate::SubdiagnosticMessage =
                    crate::SubdiagnosticMessage::FluentAttr(std::borrow::Cow::Borrowed("help"));
                pub const note: crate::SubdiagnosticMessage =
                    crate::SubdiagnosticMessage::FluentAttr(std::borrow::Cow::Borrowed("note"));
                pub const warn: crate::SubdiagnosticMessage =
                    crate::SubdiagnosticMessage::FluentAttr(std::borrow::Cow::Borrowed("warn"));
                pub const label: crate::SubdiagnosticMessage =
                    crate::SubdiagnosticMessage::FluentAttr(std::borrow::Cow::Borrowed("label"));
                pub const suggestion: crate::SubdiagnosticMessage =
                    crate::SubdiagnosticMessage::FluentAttr(std::borrow::Cow::Borrowed("suggestion"));
            }
        }
    }
    .into()
}

/// See [rustc_macros::fluent_messages].
pub(crate) fn fluent_messages(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let crate_name = std::env::var("CARGO_PKG_NAME")
        // If `CARGO_PKG_NAME` is missing, then we're probably running in a test, so use
        // `no_crate`.
        .unwrap_or_else(|_| "no_crate".to_string())
        .replace("rustc_", "");

    // Cannot iterate over individual messages in a bundle, so do that using the
    // `FluentResource` instead. Construct a bundle anyway to find out if there are conflicting
    // messages in the resources.
    let mut bundle = FluentBundle::new(vec![langid!("en-US")]);

    // Set of Fluent attribute names already output, to avoid duplicate type errors - any given
    // constant created for a given attribute is the same.
    let mut previous_attrs = HashSet::new();

    let resource_str = parse_macro_input!(input as LitStr);
    let resource_span = resource_str.span().unwrap();
    let relative_ftl_path = resource_str.value();
    let absolute_ftl_path = invocation_relative_path_to_absolute(resource_span, &relative_ftl_path);

    let crate_name = Ident::new(&crate_name, resource_str.span());

    // As this macro also outputs an `include_str!` for this file, the macro will always be
    // re-executed when the file changes.
    let mut resource_file = match File::open(absolute_ftl_path) {
        Ok(resource_file) => resource_file,
        Err(e) => {
            Diagnostic::spanned(resource_span, Level::Error, "could not open Fluent resource")
                .note(e.to_string())
                .emit();
            return failed(&crate_name);
        }
    };
    let mut resource_contents = String::new();
    if let Err(e) = resource_file.read_to_string(&mut resource_contents) {
        Diagnostic::spanned(resource_span, Level::Error, "could not read Fluent resource")
            .note(e.to_string())
            .emit();
        return failed(&crate_name);
    }
    let mut bad = false;
    for esc in ["\\n", "\\\"", "\\'"] {
        for _ in resource_contents.matches(esc) {
            bad = true;
            Diagnostic::spanned(resource_span, Level::Error, format!("invalid escape `{esc}` in Fluent resource"))
                .note("Fluent does not interpret these escape sequences (<https://projectfluent.org/fluent/guide/special.html>)")
                .emit();
        }
    }
    if bad {
        return failed(&crate_name);
    }

    let resource = match FluentResource::try_new(resource_contents) {
        Ok(resource) => resource,
        Err((this, errs)) => {
            Diagnostic::spanned(resource_span, Level::Error, "could not parse Fluent resource")
                .help("see additional errors emitted")
                .emit();
            for ParserError { pos, slice: _, kind } in errs {
                let mut err = kind.to_string();
                // Entirely unnecessary string modification so that the error message starts
                // with a lowercase as rustc errors do.
                err.replace_range(0..1, &err.chars().next().unwrap().to_lowercase().to_string());

                let line_starts: Vec<usize> = std::iter::once(0)
                    .chain(
                        this.source()
                            .char_indices()
                            .filter_map(|(i, c)| Some(i + 1).filter(|_| c == '\n')),
                    )
                    .collect();
                let line_start = line_starts
                    .iter()
                    .enumerate()
                    .map(|(line, idx)| (line + 1, idx))
                    .filter(|(_, idx)| **idx <= pos.start)
                    .last()
                    .unwrap()
                    .0;

                let snippet = Snippet {
                    title: Some(Annotation {
                        label: Some(&err),
                        id: None,
                        annotation_type: AnnotationType::Error,
                    }),
                    footer: vec![],
                    slices: vec![Slice {
                        source: this.source(),
                        line_start,
                        origin: Some(&relative_ftl_path),
                        fold: true,
                        annotations: vec![SourceAnnotation {
                            label: "",
                            annotation_type: AnnotationType::Error,
                            range: (pos.start, pos.end - 1),
                        }],
                    }],
                    opt: Default::default(),
                };
                let dl = DisplayList::from(snippet);
                eprintln!("{dl}\n");
            }

            return failed(&crate_name);
        }
    };

    let mut constants = TokenStream::new();
    let mut previous_defns = HashMap::new();
    let mut message_refs = Vec::new();
    for entry in resource.entries() {
        if let Entry::Message(Message { id: Identifier { name }, attributes, value, .. }) = entry {
            let _ = previous_defns.entry(name.to_string()).or_insert(resource_span);
            if name.contains('-') {
                Diagnostic::spanned(
                    resource_span,
                    Level::Error,
                    format!("name `{name}` contains a '-' character"),
                )
                .help("replace any '-'s with '_'s")
                .emit();
            }

            if let Some(Pattern { elements }) = value {
                for elt in elements {
                    if let PatternElement::Placeable {
                        expression:
                            Expression::Inline(InlineExpression::MessageReference { id, .. }),
                    } = elt
                    {
                        message_refs.push((id.name, *name));
                    }
                }
            }

            // `typeck_foo_bar` => `foo_bar` (in `typeck.ftl`)
            // `const_eval_baz` => `baz` (in `const_eval.ftl`)
            // `const-eval-hyphen-having` => `hyphen_having` (in `const_eval.ftl`)
            // The last case we error about above, but we want to fall back gracefully
            // so that only the error is being emitted and not also one about the macro
            // failing.
            let crate_prefix = format!("{crate_name}_");

            let snake_name = name.replace('-', "_");
            if !snake_name.starts_with(&crate_prefix) {
                Diagnostic::spanned(
                    resource_span,
                    Level::Error,
                    format!("name `{name}` does not start with the crate name"),
                )
                .help(format!(
                    "prepend `{crate_prefix}` to the slug name: `{crate_prefix}{snake_name}`"
                ))
                .emit();
            };
            let snake_name = Ident::new(&snake_name, resource_str.span());

            if !previous_attrs.insert(snake_name.clone()) {
                continue;
            }

            let msg = format!("Constant referring to Fluent message `{name}` from `{crate_name}`");
            constants.extend(quote! {
                #[doc = #msg]
                pub const #snake_name: crate::DiagnosticMessage =
                    crate::DiagnosticMessage::FluentIdentifier(
                        std::borrow::Cow::Borrowed(#name),
                        None
                    );
            });

            for Attribute { id: Identifier { name: attr_name }, .. } in attributes {
                let snake_name = Ident::new(
                    &format!("{}{}", &crate_prefix, &attr_name.replace('-', "_")),
                    resource_str.span(),
                );
                if !previous_attrs.insert(snake_name.clone()) {
                    continue;
                }

                if attr_name.contains('-') {
                    Diagnostic::spanned(
                        resource_span,
                        Level::Error,
                        format!("attribute `{attr_name}` contains a '-' character"),
                    )
                    .help("replace any '-'s with '_'s")
                    .emit();
                }

                let msg = format!(
                    "Constant referring to Fluent message `{name}.{attr_name}` from `{crate_name}`"
                );
                constants.extend(quote! {
                    #[doc = #msg]
                    pub const #snake_name: crate::SubdiagnosticMessage =
                        crate::SubdiagnosticMessage::FluentAttr(
                            std::borrow::Cow::Borrowed(#attr_name)
                        );
                });
            }
        }
    }

    for (mref, name) in message_refs.into_iter() {
        if !previous_defns.contains_key(mref) {
            Diagnostic::spanned(
                resource_span,
                Level::Error,
                format!("referenced message `{mref}` does not exist (in message `{name}`)"),
            )
            .help(&format!("you may have meant to use a variable reference (`{{${mref}}}`)"))
            .emit();
        }
    }

    if let Err(errs) = bundle.add_resource(resource) {
        for e in errs {
            match e {
                FluentError::Overriding { kind, id } => {
                    Diagnostic::spanned(
                        resource_span,
                        Level::Error,
                        format!("overrides existing {kind}: `{id}`"),
                    )
                    .emit();
                }
                FluentError::ResolverError(_) | FluentError::ParserError(_) => unreachable!(),
            }
        }
    }

    quote! {
        /// Raw content of Fluent resource for this crate, generated by `fluent_messages` macro,
        /// imported by `rustc_driver` to include all crates' resources in one bundle.
        pub static DEFAULT_LOCALE_RESOURCE: &'static str = include_str!(#relative_ftl_path);

        #[allow(non_upper_case_globals)]
        #[doc(hidden)]
        /// Auto-generated constants for type-checked references to Fluent messages.
        pub(crate) mod fluent_generated {
            #constants

            /// Constants expected to exist by the diagnostic derive macros to use as default Fluent
            /// identifiers for different subdiagnostic kinds.
            pub mod _subdiag {
                /// Default for `#[help]`
                pub const help: crate::SubdiagnosticMessage =
                    crate::SubdiagnosticMessage::FluentAttr(std::borrow::Cow::Borrowed("help"));
                /// Default for `#[note]`
                pub const note: crate::SubdiagnosticMessage =
                    crate::SubdiagnosticMessage::FluentAttr(std::borrow::Cow::Borrowed("note"));
                /// Default for `#[warn]`
                pub const warn: crate::SubdiagnosticMessage =
                    crate::SubdiagnosticMessage::FluentAttr(std::borrow::Cow::Borrowed("warn"));
                /// Default for `#[label]`
                pub const label: crate::SubdiagnosticMessage =
                    crate::SubdiagnosticMessage::FluentAttr(std::borrow::Cow::Borrowed("label"));
                /// Default for `#[suggestion]`
                pub const suggestion: crate::SubdiagnosticMessage =
                    crate::SubdiagnosticMessage::FluentAttr(std::borrow::Cow::Borrowed("suggestion"));
            }
        }
    }
    .into()
}
