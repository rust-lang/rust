//rustfmt-format_macro_bodies: true

macro_rules! mto_text_left {
    ($buf:ident, $n:ident, $pos:ident, $state:ident) => {{
        let cursor = loop {
            state = match iter.next() {
                None if $pos == DP::Start => break last_char_idx($buf),
                None /*some comment */ => break 0,
            };
        };
        Ok(saturate_cursor($buf, cursor))
    }};
}
