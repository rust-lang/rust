use std::borrow::Cow;

pub use rustc_error_messages::FluentArgs;
use rustc_error_messages::{DiagArgMap, langid, register_functions};
use tracing::{debug, trace};

use crate::fluent_bundle::FluentResource;
use crate::{DiagArg, DiagMessage, Style, fluent_bundle};

/// Convert diagnostic arguments (a rustc internal type that exists to implement
/// `Encodable`/`Decodable`) into `FluentArgs` which is necessary to perform translation.
///
/// Typically performed once for each diagnostic at the start of `emit_diagnostic` and then
/// passed around as a reference thereafter.
fn to_fluent_args<'iter>(iter: impl Iterator<Item = DiagArg<'iter>>) -> FluentArgs<'static> {
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

/// Convert `DiagMessage`s to a string
pub fn format_diag_messages(
    messages: &[(DiagMessage, Style)],
    args: &DiagArgMap,
) -> Cow<'static, str> {
    Cow::Owned(messages.iter().map(|(m, _)| format_diag_message(m, args)).collect::<String>())
}

/// Convert a `DiagMessage` to a string
pub fn format_diag_message<'a>(message: &'a DiagMessage, args: &DiagArgMap) -> Cow<'a, str> {
    trace!(?message, ?args);

    match message {
        DiagMessage::Str(msg) => Cow::Borrowed(msg),
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
            let args = to_fluent_args(args.iter());

            let mut errs = vec![];
            let translated = bundle.format_pattern(value, Some(&args), &mut errs).to_string();
            debug!(?translated, ?errs);
            if errs.is_empty() {
                Cow::Owned(translated)
            } else {
                panic!("Fluent errors while formatting message: {errs:?}");
            }
        }
    }
}
