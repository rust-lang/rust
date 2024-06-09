// FIXME: document that the "channel" is indeed a channel *URL*!

use std::{borrow::Cow, sync::OnceLock};

use crate::error::DiagCtxt;

const PLACEHOLDER: &str = "{{channel}}";
const ENV_VAR_KEY: &str = "DOC_RUST_LANG_ORG_CHANNEL";

pub(crate) fn instantiate<'a>(input: &'a str, dcx: &mut DiagCtxt) -> Result<Cow<'a, str>, ()> {
    let Some(channel) = channel(dcx)? else { return Ok(input.into()) };
    Ok(input.replace(PLACEHOLDER, channel).into())
}

#[allow(dead_code)] // FIXME
pub(crate) fn anonymize<'a>(input: &'a str, dcx: &'_ mut DiagCtxt) -> Result<Cow<'a, str>, ()> {
    let Some(channel) = channel(dcx)? else { return Ok(input.into()) };
    Ok(input.replace(channel, PLACEHOLDER).into())
}

fn channel(dcx: &mut DiagCtxt) -> Result<Option<&'static str>, ()> {
    static CHANNEL_URL: OnceLock<Option<String>> = OnceLock::new();

    // FIXME: Use `get_or_try_init` here (instead of `get`→`set`→`get`) if/once stabilized (on beta).

    if let Some(channel_url) = CHANNEL_URL.get() {
        return Ok(channel_url.as_deref());
    }

    let channel_url = match std::env::var(ENV_VAR_KEY) {
        Ok(url) => Some(url),
        // FIXME: should we make the channel mandatory instead?
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(var)) => {
            // FIXME: better diag
            // FIXME: Use `OsStr::display` (instead of `to_string_lossy`) if/once stabilized (on beta).
            dcx.emit(
                &format!("env var `{ENV_VAR_KEY}` is not valid UTF-8: `{}`", var.to_string_lossy()),
                None,
                None,
            );
            return Err(());
        }
    };

    // unwrap: The static item is locally scoped and no other thread tries to initialize it.
    CHANNEL_URL.set(channel_url).unwrap();
    // unwrap: Initialized above.
    Ok(CHANNEL_URL.get().unwrap().as_deref())
}
