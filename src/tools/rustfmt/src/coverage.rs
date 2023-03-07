use crate::{Config, EmitMode};
use std::borrow::Cow;

pub(crate) fn transform_missing_snippet<'a>(config: &Config, string: &'a str) -> Cow<'a, str> {
    match config.emit_mode() {
        EmitMode::Coverage => Cow::from(replace_chars(string)),
        _ => Cow::from(string),
    }
}

fn replace_chars(s: &str) -> String {
    s.chars()
        .map(|ch| if ch.is_whitespace() { ch } else { 'X' })
        .collect()
}
