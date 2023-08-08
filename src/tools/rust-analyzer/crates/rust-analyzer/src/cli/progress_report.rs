//! A simple progress bar
//!
//! A single thread non-optimized progress bar
use std::io::{self, Write};

/// A Simple ASCII Progress Bar
pub(crate) struct ProgressReport<'a> {
    curr: f32,
    text: String,
    hidden: bool,

    len: u64,
    pos: u64,
    msg: Option<Box<dyn Fn() -> String + 'a>>,
}

impl<'a> ProgressReport<'a> {
    pub(crate) fn new(len: u64) -> ProgressReport<'a> {
        ProgressReport { curr: 0.0, text: String::new(), hidden: false, len, pos: 0, msg: None }
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
            self.msg.as_ref().map_or_else(|| String::new(), |it| it())
        );
        self.update_text(&text);
    }

    fn update_text(&mut self, text: &str) {
        // Get length of common portion
        let mut common_prefix_length = 0;
        let common_length = usize::min(self.text.len(), text.len());

        while common_prefix_length < common_length
            && text.chars().nth(common_prefix_length).unwrap()
                == self.text.chars().nth(common_prefix_length).unwrap()
        {
            common_prefix_length += 1;
        }

        // Backtrack to the first differing character
        let mut output = String::new();
        output += &'\x08'.to_string().repeat(self.text.len() - common_prefix_length);
        // Output new suffix
        output += &text[common_prefix_length..text.len()];

        // If the new text is shorter than the old one: delete overlapping characters
        if let Some(overlap_count) = self.text.len().checked_sub(text.len()) {
            if overlap_count > 0 {
                output += &" ".repeat(overlap_count);
                output += &"\x08".repeat(overlap_count);
            }
        }

        let _ = io::stdout().write(output.as_bytes());
        let _ = io::stdout().flush();
        self.text = text.to_string();
    }

    fn set_value(&mut self, value: f32) {
        self.curr = f32::max(0.0, f32::min(1.0, value));
    }

    fn clear(&mut self) {
        if self.hidden {
            return;
        }

        // Fill all last text to space and return the cursor
        let spaces = " ".repeat(self.text.len());
        let backspaces = "\x08".repeat(self.text.len());
        print!("{backspaces}{spaces}{backspaces}");
        let _ = io::stdout().flush();

        self.text = String::new();
    }
}
