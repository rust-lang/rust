use std::str::pattern::{Pattern, SearchStep, Searcher};

use crate::re_unicode::{Matches, Regex};

#[derive(Debug)]
pub struct RegexSearcher<'r, 't> {
    haystack: &'t str,
    it: Matches<'r, 't>,
    last_step_end: usize,
    next_match: Option<(usize, usize)>,
}

impl<'r, 't> Pattern<'t> for &'r Regex {
    type Searcher = RegexSearcher<'r, 't>;

    fn into_searcher(self, haystack: &'t str) -> RegexSearcher<'r, 't> {
        RegexSearcher {
            haystack,
            it: self.find_iter(haystack),
            last_step_end: 0,
            next_match: None,
        }
    }
}

unsafe impl<'r, 't> Searcher<'t> for RegexSearcher<'r, 't> {
    #[inline]
    fn haystack(&self) -> &'t str {
        self.haystack
    }

    #[inline]
    fn next(&mut self) -> SearchStep {
        if let Some((s, e)) = self.next_match {
            self.next_match = None;
            self.last_step_end = e;
            return SearchStep::Match(s, e);
        }
        match self.it.next() {
            None => {
                if self.last_step_end < self.haystack().len() {
                    let last = self.last_step_end;
                    self.last_step_end = self.haystack().len();
                    SearchStep::Reject(last, self.haystack().len())
                } else {
                    SearchStep::Done
                }
            }
            Some(m) => {
                let (s, e) = (m.start(), m.end());
                if s == self.last_step_end {
                    self.last_step_end = e;
                    SearchStep::Match(s, e)
                } else {
                    self.next_match = Some((s, e));
                    let last = self.last_step_end;
                    self.last_step_end = s;
                    SearchStep::Reject(last, s)
                }
            }
        }
    }
}
