use crate::snippet::Style;
use crate::{DiagnosticArg, DiagnosticMessage, FluentBundle};
use rustc_data_structures::sync::Lrc;
use rustc_error_messages::FluentArgs;
use std::borrow::Cow;

pub trait Translate {
    /// Return `FluentBundle` with localized diagnostics for the locale requested by the user. If no
    /// language was requested by the user then this will be `None` and `fallback_fluent_bundle`
    /// should be used.
    fn fluent_bundle(&self) -> Option<&Lrc<FluentBundle>>;

    /// Return `FluentBundle` with localized diagnostics for the default locale of the compiler.
    /// Used when the user has not requested a specific language or when a localized diagnostic is
    /// unavailable for the requested locale.
    fn fallback_fluent_bundle(&self) -> &FluentBundle;

    /// Convert diagnostic arguments (a rustc internal type that exists to implement
    /// `Encodable`/`Decodable`) into `FluentArgs` which is necessary to perform translation.
    ///
    /// Typically performed once for each diagnostic at the start of `emit_diagnostic` and then
    /// passed around as a reference thereafter.
    fn to_fluent_args<'arg>(&self, args: &[DiagnosticArg<'arg>]) -> FluentArgs<'arg> {
        FromIterator::from_iter(args.iter().cloned())
    }

    /// Convert `DiagnosticMessage`s to a string, performing translation if necessary.
    fn translate_messages(
        &self,
        messages: &[(DiagnosticMessage, Style)],
        args: &FluentArgs<'_>,
    ) -> Cow<'_, str> {
        Cow::Owned(
            messages.iter().map(|(m, _)| self.translate_message(m, args)).collect::<String>(),
        )
    }

    /// Convert a `DiagnosticMessage` to a string, performing translation if necessary.
    fn translate_message<'a>(
        &'a self,
        message: &'a DiagnosticMessage,
        args: &'a FluentArgs<'_>,
    ) -> Cow<'_, str> {
        trace!(?message, ?args);
        let (identifier, attr) = match message {
            DiagnosticMessage::Str(msg) => return Cow::Borrowed(&msg),
            DiagnosticMessage::FluentIdentifier(identifier, attr) => (identifier, attr),
        };

        let translate_with_bundle = |bundle: &'a FluentBundle| -> Option<(Cow<'_, str>, Vec<_>)> {
            let message = bundle.get_message(&identifier)?;
            let value = match attr {
                Some(attr) => message.get_attribute(attr)?.value(),
                None => message.value()?,
            };
            debug!(?message, ?value);

            let mut errs = vec![];
            let translated = bundle.format_pattern(value, Some(&args), &mut errs);
            debug!(?translated, ?errs);
            Some((translated, errs))
        };

        self.fluent_bundle()
            .and_then(|bundle| translate_with_bundle(bundle))
            // If `translate_with_bundle` returns `None` with the primary bundle, this is likely
            // just that the primary bundle doesn't contain the message being translated, so
            // proceed to the fallback bundle.
            //
            // However, when errors are produced from translation, then that means the translation
            // is broken (e.g. `{$foo}` exists in a translation but `foo` isn't provided).
            //
            // In debug builds, assert so that compiler devs can spot the broken translation and
            // fix it..
            .inspect(|(_, errs)| {
                debug_assert!(
                    errs.is_empty(),
                    "identifier: {:?}, attr: {:?}, args: {:?}, errors: {:?}",
                    identifier,
                    attr,
                    args,
                    errs
                );
            })
            // ..otherwise, for end users, an error about this wouldn't be useful or actionable, so
            // just hide it and try with the fallback bundle.
            .filter(|(_, errs)| errs.is_empty())
            .or_else(|| translate_with_bundle(self.fallback_fluent_bundle()))
            .map(|(translated, errs)| {
                // Always bail out for errors with the fallback bundle.
                assert!(
                    errs.is_empty(),
                    "identifier: {:?}, attr: {:?}, args: {:?}, errors: {:?}",
                    identifier,
                    attr,
                    args,
                    errs
                );
                translated
            })
            .expect("failed to find message in primary or fallback fluent bundles")
    }
}
