//! A simple progress bar
//!
//! A single thread non-optimized progress bar
use std::io::{self, IsTerminal, Write};

/// A Simple ASCII Progress Bar
pub(crate) struct ProgressReport<'a> {
    curr: f32,
    text: String,
    hidden: bool,

    len: usize,
    pos: u64,
    msg: Option<Box<dyn Fn() -> String + 'a>>,
}

impl<'a> ProgressReport<'a> {
    pub(crate) fn new(len: usize) -> ProgressReport<'a> {
        let is_tty = io::stdout().is_terminal();
        ProgressReport { curr: 0.0, text: String::new(), hidden: !is_tty, len, pos: 0, msg: None }
    }

    pub(crate) fn hidden() -> ProgressReport<'a> {
        ProgressReport { curr: 0.0, text: String::new(), hidden: true, len: 0, pos: 0, msg: None }
    }

    pub(crate) fn set_message(&mut self, msg: impl Fn() -> String + 'a) {
        if !self.hidden {
            self.msg = Some(Box::new(msg));
        }
        self.tick();
    }

    pub(crate) fn println<I: Into<String>>(&mut self, msg: I) {
        self.clear();
        println!("{}", msg.into());
        self.tick();
    }

    pub(crate) fn inc(&mut self, delta: u64) {
        self.pos += delta;
        if self.len == 0 {
            self.set_value(0.0)
        } else {
            self.set_value((self.pos as f32) / (self.len as f32))
        }
        self.tick();
    }

    pub(crate) fn finish_and_clear(&mut self) {
        self.clear();
    }

    pub(crate) fn tick(&mut self) {
        if self.hidden {
            return;
        }
        let percent = (self.curr * 100.0) as u32;
        let text = format!(
            "{}/{} {percent:3>}% {}",
            self.pos,
            self.len,
            self.msg.as_ref().map_or_else(String::new, |it| it())
        );
        self.update_text(&text);
    }

    fn update_text(&mut self, text: &str) {
        let output = render_text_update(&self.text, text);
        let _ = io::stdout().write(output.as_bytes());
        let _ = io::stdout().flush();
        text.clone_into(&mut self.text);
    }

    fn set_value(&mut self, value: f32) {
        self.curr = value.clamp(0.0, 1.0);
    }

    fn clear(&mut self) {
        if self.hidden {
            return;
        }

        // Fill all last text to space and return the cursor
        let len = self.text.chars().count();
        let spaces = " ".repeat(len);
        let backspaces = "\x08".repeat(len);
        print!("{backspaces}{spaces}{backspaces}");
        let _ = io::stdout().flush();

        self.text = String::new();
    }
}

fn render_text_update(old: &str, new: &str) -> String {
    let old_len = old.chars().count();
    let new_len = new.chars().count();

    // Get length of common portion
    let mut common_prefix_length = 0;
    let common_length = usize::min(old_len, new_len);

    while common_prefix_length < common_length
        && new.chars().nth(common_prefix_length).unwrap()
            == old.chars().nth(common_prefix_length).unwrap()
    {
        common_prefix_length += 1;
    }

    // Backtrack to the first differing character
    let mut output = String::new();
    output += &'\x08'.to_string().repeat(old_len - common_prefix_length);
    // Output new suffix, using chars() iter to ensure unicode compatibility
    output.extend(new.chars().skip(common_prefix_length));

    // If the new text is shorter than the old one: delete overlapping characters
    if let Some(overlap_count) = old_len.checked_sub(new_len)
        && overlap_count > 0
    {
        output += &" ".repeat(overlap_count);
        output += &"\x08".repeat(overlap_count);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::render_text_update;

    #[test]
    fn ascii_prefix_reuse() {
        let old = "1/7 14% processing: foo";
        let new = "1/7 28% processing: bar";
        let update = render_text_update(old, new);

        let common = "1/7 ";
        let backspaces = old.chars().count() - common.chars().count();
        let expected = format!("{}{}", "\x08".repeat(backspaces), "28% processing: bar");
        assert_eq!(update, expected);
    }

    #[test]
    fn unicode_identifiers_do_not_panic() {
        // Regression test for rust-lang/rust-analyzer#22844: previous code
        // compared byte lengths with char indices, so `chars().nth(...).unwrap()`
        // panicked on non-ASCII.
        let old = "1/7 14% processing: f::消息";
        let new = "2/7 28% processing: f::消息内容";
        let update = render_text_update(old, new);

        let backspaces = old.chars().count();
        let expected = format!("{}{new}", "\x08".repeat(backspaces));
        assert_eq!(update, expected);
    }

    #[test]
    fn shorter_unicode_message_clears_overlap() {
        let old = "processing: 消息内容";
        let new = "processing: 消息";
        let update = render_text_update(old, new);

        // Drop the last two chars, then blank/backspace the leftover width.
        let expected = "\x08\x08  \x08\x08";
        assert_eq!(update, expected);
    }
}
