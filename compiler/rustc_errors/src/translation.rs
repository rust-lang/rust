use std::borrow::Cow;
use std::error::Report;
use std::sync::Arc;

pub use rustc_error_messages::{FluentArgs, LazyFallbackBundle};
use rustc_error_messages::{langid, register_functions};
use tracing::{debug, trace};

use crate::error::TranslateError;
use crate::fluent_bundle::FluentResource;
use crate::{DiagArg, DiagMessage, FluentBundle, Style, fluent_bundle};

/// Convert diagnostic arguments (a rustc internal type that exists to implement
/// `Encodable`/`Decodable`) into `FluentArgs` which is necessary to perform translation.
///
/// Typically performed once for each diagnostic at the start of `emit_diagnostic` and then
/// passed around as a reference thereafter.
pub fn to_fluent_args<'iter>(iter: impl Iterator<Item = DiagArg<'iter>>) -> FluentArgs<'static> {
    let mut args = if let Some(size) = iter.size_hint().1 {
        FluentArgs::with_capacity(size)
    } else {
        FluentArgs::new()
    };

    for (k, v) in iter {
        args.set(k.clone(), v.clone());
    }

    args
}

#[derive(Clone)]
pub struct Translator {
    /// Localized diagnostics for the locale requested by the user. If no language was requested by
    /// the user then this will be `None` and `fallback_fluent_bundle` should be used.
    pub fluent_bundle: Option<Arc<FluentBundle>>,
}

impl Translator {
    pub fn new() -> Translator {
        Translator { fluent_bundle: None }
    }

    /// Convert `DiagMessage`s to a string, performing translation if necessary.
    pub fn translate_messages(
        &self,
        messages: &[(DiagMessage, Style)],
        args: &FluentArgs<'_>,
    ) -> Cow<'_, str> {
        Cow::Owned(
            messages
                .iter()
                .map(|(m, _)| self.translate_message(m, args).map_err(Report::new).unwrap())
                .collect::<String>(),
        )
    }

    /// Convert a `DiagMessage` to a string, performing translation if necessary.
    pub fn translate_message<'a>(
        &'a self,
        message: &'a DiagMessage,
        args: &'a FluentArgs<'_>,
    ) -> Result<Cow<'a, str>, TranslateError<'a>> {
        trace!(?message, ?args);
        match message {
            DiagMessage::Str(msg) => Ok(Cow::Borrowed(msg)),
            // This translates an inline fluent diagnostic message
            // It does this by creating a new `FluentBundle` with only one message,
            // and then translating using this bundle.
            DiagMessage::Inline(msg) => {
                const GENERATED_MSG_ID: &str = "generated_msg";
                let resource =
                    FluentResource::try_new(format!("{GENERATED_MSG_ID} = {msg}\n")).unwrap();
                let mut bundle = fluent_bundle::FluentBundle::new(vec![langid!("en-US")]);
                bundle.set_use_isolating(false);
                bundle.add_resource(resource).unwrap();
                register_functions(&mut bundle);
                let message = bundle.get_message(GENERATED_MSG_ID).unwrap();
                let value = message.value().unwrap();

                let mut errs = vec![];
                let translated = bundle.format_pattern(value, Some(args), &mut errs).to_string();
                debug!(?translated, ?errs);
                if errs.is_empty() {
                    Ok(Cow::Owned(translated))
                } else {
                    Err(TranslateError::fluent(&Cow::Borrowed(GENERATED_MSG_ID), args, errs))
                }
            }
        }
    }
}
