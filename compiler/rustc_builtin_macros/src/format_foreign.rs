pub(crate) mod printf;

pub(crate) mod shell;

#[derive(Clone, Copy)]
struct StrCursor<'a> {
    s: &'a str,
    pub at: usize,
}

impl<'a> StrCursor<'a> {
    fn new_at(s: &'a str, at: usize) -> StrCursor<'a> {
        StrCursor { s, at }
    }

    fn at_next_cp(mut self) -> Option<StrCursor<'a>> {
        match self.try_seek_right_cp() {
            true => Some(self),
            false => None,
        }
    }

    fn next_cp(mut self) -> Option<(char, StrCursor<'a>)> {
        let cp = self.cp_after()?;
        self.seek_right(cp.len_utf8());
        Some((cp, self))
    }

    fn slice_before(&self) -> &'a str {
        &self.s[0..self.at]
    }

    fn slice_after(&self) -> &'a str {
        &self.s[self.at..]
    }

    fn slice_between(&self, until: StrCursor<'a>) -> Option<&'a str> {
        if !str_eq_literal(self.s, until.s) {
            None
        } else {
            use std::cmp::{max, min};
            let beg = min(self.at, until.at);
            let end = max(self.at, until.at);
            Some(&self.s[beg..end])
        }
    }

    fn cp_after(&self) -> Option<char> {
        self.slice_after().chars().next()
    }

    fn try_seek_right_cp(&mut self) -> bool {
        match self.slice_after().chars().next() {
            Some(c) => {
                self.at += c.len_utf8();
                true
            }
            None => false,
        }
    }

    fn seek_right(&mut self, bytes: usize) {
        self.at += bytes;
    }
}

impl std::fmt::Debug for StrCursor<'_> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(fmt, "StrCursor({:?} | {:?})", self.slice_before(), self.slice_after())
    }
}

fn str_eq_literal(a: &str, b: &str) -> bool {
    a.as_bytes().as_ptr() == b.as_bytes().as_ptr() && a.len() == b.len()
}
