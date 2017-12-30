use {TextUnit};

use std::str::Chars;

pub(crate) struct Ptr<'s> {
    text: &'s str,
    len: TextUnit,
}

impl<'s> Ptr<'s> {
    pub fn new(text: &'s str) -> Ptr<'s> {
        Ptr { text, len: TextUnit::new(0) }
    }

    pub fn into_len(self) -> TextUnit {
        self.len
    }

    pub fn next(&self) -> Option<char> {
        self.chars().next()
    }

    pub fn nnext(&self) -> Option<char> {
        let mut chars = self.chars();
        chars.next()?;
        chars.next()
    }

    pub fn next_is(&self, c: char) -> bool {
        self.next() == Some(c)
    }

    pub fn nnext_is(&self, c: char) -> bool {
        self.nnext() == Some(c)
    }

    pub fn nnext_is_p<P: Fn(char) -> bool>(&self, p: P) -> bool {
        self.nnext().map(p) == Some(true)
    }

    pub fn bump(&mut self) -> Option<char> {
        let ch = self.chars().next()?;
        self.len += TextUnit::len_of_char(ch);
        Some(ch)
    }

    pub fn bump_while<F: Fn(char) -> bool>(&mut self, pred: F) {
        loop {
            match self.next() {
                Some(c) if pred(c) => {
                    self.bump();
                },
                _ => return,
            }
        }
    }

    fn chars(&self) -> Chars {
        let len: u32 = self.len.into();
        self.text[len as usize ..].chars()
    }
}
