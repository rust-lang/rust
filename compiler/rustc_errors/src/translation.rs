use crate::error::{TranslateError, TranslateErrorKind};
use crate::snippet::Style;
use crate::{DiagnosticArg, DiagnosticMessage};
use rustc_data_structures::sync::Lrc;
use rustc_error_messages::fluent_bundle::FluentResource;
pub use rustc_error_messages::FluentArgs;
use rustc_error_messages::{langid, new_bundle, FluentBundle};
use std::borrow::Cow;
use std::env;
use std::error::Report;

#[cfg(not(parallel_compiler))]
use std::cell::LazyCell as Lazy;
#[cfg(parallel_compiler)]
use std::sync::LazyLock as Lazy;

/// Convert diagnostic arguments (a rustc internal type that exists to implement
/// `Encodable`/`Decodable`) into `FluentArgs` which is necessary to perform translation.
///
/// Typically performed once for each diagnostic at the start of `emit_diagnostic` and then
/// passed around as a reference thereafter.
pub fn to_fluent_args<'iter>(
    iter: impl Iterator<Item = DiagnosticArg<'iter>>,
) -> FluentArgs<'static> {
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

pub trait Translate {
    /// Return `FluentBundle` with localized diagnostics for the locale requested by the user. If no
    /// language was requested by the user then this will be `None` and `fallback_fluent_bundle`
    /// should be used.
    fn fluent_bundle(&self) -> Option<&Lrc<FluentBundle>>;

    /// Return `FluentBundle` with localized diagnostics for the default locale of the compiler.
    /// Used when the user has not requested a specific language or when a localized diagnostic is
    /// unavailable for the requested locale.
    fn fallback_fluent_bundle(&self) -> &FluentBundle;

    /// Convert `DiagnosticMessage`s to a string, performing translation if necessary.
    fn translate_messages(
        &self,
        messages: &[(DiagnosticMessage, Style)],
        args: &FluentArgs<'_>,
    ) -> Cow<'_, str> {
        Cow::Owned(
            messages
                .iter()
                .map(|(m, _)| self.translate_message(m, args).map_err(Report::new).unwrap())
                .collect::<String>(),
        )
    }

    /// Convert a `DiagnosticMessage` to a string, performing translation if necessary.
    fn translate_message<'a>(
        &'a self,
        message: &'a DiagnosticMessage,
        args: &'a FluentArgs<'_>,
    ) -> Result<Cow<'_, str>, TranslateError<'_>> {
        let fallback = |translator: &'a Self| translator.fallback_fluent_bundle();
        let (identifier, attr) = match message {
            DiagnosticMessage::Str(msg) | DiagnosticMessage::Translated(msg) => {
                return Ok(Cow::Borrowed(msg));
            }
            DiagnosticMessage::FluentIdentifier(identifier, attr) => (identifier, attr),
        };
        translate_message(self, fallback, identifier, attr.as_ref(), args)
    }

    /// Translate a raw Fluent string.
    fn raw_translate_message<'t: 'msg, 'msg>(
        &'t self,
        slug: &'msg str,
        raw: &'msg str,
        args: &'msg FluentArgs<'_>,
    ) -> String {
        let fallback = Lazy::new(move || {
            let res = FluentResource::try_new(format!("{slug} = {raw}"))
                .expect("failed to parse fallback fluent resource");
            let mut bundle = new_bundle(vec![langid!("en-US")]);
            // FIXME(davidtwco): get this value from the option
            bundle.set_use_isolating(false);
            bundle.add_resource(res).expect("adding resource");
            bundle
        });
        let fallback = |_| &*fallback;
        let slug = Cow::Borrowed(slug);
        let translated = translate_message(self, fallback, &slug, None, args);
        translated.map_err(Report::new).unwrap().to_string()
    }
}

fn translate_message<'t: 'bundle, 'bundle: 'msg, 'msg, T: Translate + ?Sized>(
    translator: &'t T,
    fallback: impl Fn(&'t T) -> &'bundle FluentBundle,
    identifier: &'msg Cow<'msg, str>,
    attr: Option<&'msg Cow<'msg, str>>,
    args: &'msg FluentArgs<'msg>,
) -> Result<Cow<'msg, str>, TranslateError<'msg>> {
    let translate_with_bundle =
        |bundle: &'bundle FluentBundle| -> Result<Cow<'msg, str>, TranslateError<'msg>> {
            let message =
                bundle.get_message(identifier).ok_or(TranslateError::message(identifier, args))?;
            let value = match attr {
                Some(attr) => message
                    .get_attribute(attr)
                    .ok_or(TranslateError::attribute(identifier, args, attr))?
                    .value(),
                None => message.value().ok_or(TranslateError::value(identifier, args))?,
            };
            debug!(?message, ?value);

            let mut errs = vec![];
            let translated = bundle.format_pattern(value, Some(args), &mut errs);
            debug!(?translated, ?errs);
            if errs.is_empty() {
                Ok(translated)
            } else {
                Err(TranslateError::fluent(identifier, args, errs))
            }
        };

    try {
        match translator.fluent_bundle().map(|b| translate_with_bundle(b)) {
            // The primary bundle was present and translation succeeded
            Some(Ok(t)) => t,

            // If `translate_with_bundle` returns `Err` with the primary bundle, this is likely
            // just that the primary bundle doesn't contain the message being translated, so
            // proceed to the fallback bundle.
            Some(Err(
                primary @ TranslateError::One { kind: TranslateErrorKind::MessageMissing, .. },
            )) => translate_with_bundle(fallback(translator))
                .map_err(|fallback| primary.and(fallback))?,

            // Always yeet out for errors on debug (unless
            // `RUSTC_TRANSLATION_NO_DEBUG_ASSERT` is set in the environment - this allows
            // local runs of the test suites, of builds with debug assertions, to test the
            // behaviour in a normal build).
            Some(Err(primary))
                if cfg!(debug_assertions)
                    && env::var("RUSTC_TRANSLATION_NO_DEBUG_ASSERT").is_err() =>
            {
                do yeet primary
            }

            // ..otherwise, for end users, an error about this wouldn't be useful or actionable, so
            // just hide it and try with the fallback bundle.
            Some(Err(primary)) => translate_with_bundle(fallback(translator))
                .map_err(|fallback| primary.and(fallback))?,

            // The primary bundle is missing, proceed to the fallback bundle
            None => translate_with_bundle(fallback(translator))
                .map_err(|fallback| TranslateError::primary(identifier, args).and(fallback))?,
        }
    }
}
