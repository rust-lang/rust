use std::collections::HashMap;

use proc_macro2::TokenStream;
use syn::{Attribute, Path, parse_quote};

use crate::diagnostics::error::{DiagnosticDeriveError, throw_invalid_attr};
use crate::diagnostics::generate_diag::DiagnosticDeriveKind;
use crate::diagnostics::utils::{SubdiagnosticKind, SubdiagnosticVariant};

pub(crate) struct SubdiagnosticAttribute {
    pub kind: SubdiagnosticKind,
    pub slug: Path,
}

impl DiagnosticDeriveKind {
    /// Parse a `SubdiagnosticKind` from an `Attribute`.
    pub(crate) fn parse_subdiag_attribute(
        self,
        attr: &Attribute,
        field_map: &HashMap<String, TokenStream>,
    ) -> Result<Option<SubdiagnosticAttribute>, DiagnosticDeriveError> {
        let Some(mut subdiag) = SubdiagnosticVariant::from_attr(attr, field_map)? else {
            // Some attributes aren't errors - like documentation comments - but also aren't
            // subdiagnostics.
            return Ok(None);
        };

        if let SubdiagnosticKind::MultipartSuggestion { .. } = subdiag.kind {
            throw_invalid_attr!(attr, |diag| diag
                .help("consider creating a `Subdiagnostic` instead"));
        }

        // Put in fallback slug
        let slug = subdiag.slug.take().unwrap_or(match subdiag.kind {
            SubdiagnosticKind::Label => parse_quote! { _subdiag::label },
            SubdiagnosticKind::Note => parse_quote! { _subdiag::note },
            SubdiagnosticKind::NoteOnce => parse_quote! { _subdiag::note_once },
            SubdiagnosticKind::Help => parse_quote! { _subdiag::help },
            SubdiagnosticKind::HelpOnce => parse_quote! { _subdiag::help_once },
            SubdiagnosticKind::Warn => parse_quote! { _subdiag::warn },
            SubdiagnosticKind::Suggestion { .. } => parse_quote! { _subdiag::suggestion },
            SubdiagnosticKind::MultipartSuggestion { .. } => unreachable!(),
        });

        Ok(Some(SubdiagnosticAttribute { slug, kind: subdiag.kind }))
    }
}
