// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(missing_docs)]

use prelude::*;
use super::CharEq;

// Pattern

pub trait Pattern<'a>: Sized {
    type Searcher: Searcher<'a>;
    fn into_matcher(self, haystack: &'a str) -> Self::Searcher;

    #[inline]
    fn is_contained_in(self, haystack: &'a str) -> bool {
        self.into_matcher(haystack).next_match().is_some()
    }

    #[inline]
    fn match_starts_at(self, haystack: &'a str, idx: usize) -> bool {
        let mut matcher = self.into_matcher(haystack);
        loop {
            match matcher.next() {
                SearchStep::Match(i, _) if i == idx => return true,
                SearchStep::Match(i, _)
                | SearchStep::Reject(i, _) if i >= idx => break,
                SearchStep::Done => break,
                _ => continue,
            }
        }
        false
    }

    #[inline]
    fn match_ends_at(self, haystack: &'a str, idx: usize) -> bool
    where Self::Searcher: ReverseSearcher<'a> {
        let mut matcher = self.into_matcher(haystack);
        loop {
            match matcher.next_back() {
                SearchStep::Match(_, j) if idx == j => return true,
                SearchStep::Match(_, j)
                | SearchStep::Reject(_, j) if idx >= j => break,
                SearchStep::Done => break,
                _ => continue,
            }
        }
        false
    }
}

// Searcher

pub enum SearchStep {
    Match(usize, usize),
    Reject(usize, usize),
    Done
}

pub unsafe trait Searcher<'a> {
    fn haystack(&self) -> &'a str;
    fn next(&mut self) -> SearchStep;
    #[inline]
    fn next_match(&mut self) -> Option<(usize, usize)> {
        loop {
            match self.next() {
                SearchStep::Match(a, b) => return Some((a, b)),
                SearchStep::Done => return None,
                _ => continue,
            }
        }
    }
    #[inline]
    fn next_reject(&mut self) -> Option<(usize, usize)>{
        loop {
            match self.next() {
                SearchStep::Reject(a, b) => return Some((a, b)),
                SearchStep::Done => return None,
                _ => continue,
            }
        }
    }
}

pub unsafe trait ReverseSearcher<'a>: Searcher<'a> {
    fn next_back(&mut self) -> SearchStep;
    #[inline]
    fn next_match_back(&mut self) -> Option<(usize, usize)>{
        loop {
            match self.next_back() {
                SearchStep::Match(a, b) => return Some((a, b)),
                SearchStep::Done => return None,
                _ => continue,
            }
        }
    }
    #[inline]
    fn next_reject_back(&mut self) -> Option<(usize, usize)>{
        loop {
            match self.next_back() {
                SearchStep::Reject(a, b) => return Some((a, b)),
                SearchStep::Done => return None,
                _ => continue,
            }
        }
    }
}

pub trait DoubleEndedSearcher<'a>: ReverseSearcher<'a> {}

// Impl for CharEq

pub struct CharEqSearcher<'a, C> {
    char_eq: C,
    haystack: &'a str,
    char_indices: super::CharIndices<'a>,
    #[allow(dead_code)]
    ascii_only: bool,
}

impl<'a, C: CharEq> Pattern<'a> for C {
    type Searcher = CharEqSearcher<'a, C>;

    #[inline]
    fn into_matcher(self, haystack: &'a str) -> CharEqSearcher<'a, C> {
        CharEqSearcher {
            ascii_only: self.only_ascii(),
            haystack: haystack,
            char_eq: self,
            char_indices: haystack.char_indices(),
        }
    }
}

unsafe impl<'a, C: CharEq> Searcher<'a> for CharEqSearcher<'a, C> {
    #[inline]
    fn haystack(&self) -> &'a str {
        self.haystack
    }

    #[inline]
    fn next(&mut self) -> SearchStep {
        let s = &mut self.char_indices;
        // Compare lengths of the internal byte slice iterator
        // to find length of current char
        let (pre_len, _) = s.iter.iter.size_hint();
        if let Some((i, c)) = s.next() {
            let (len, _) = s.iter.iter.size_hint();
            let char_len = pre_len - len;
            if self.char_eq.matches(c) {
                return SearchStep::Match(i, i + char_len);
            } else {
                return SearchStep::Reject(i, i + char_len);
            }
        }
        SearchStep::Done
    }
}

unsafe impl<'a, C: CharEq> ReverseSearcher<'a> for CharEqSearcher<'a, C> {
    #[inline]
    fn next_back(&mut self) -> SearchStep {
        let s = &mut self.char_indices;
        // Compare lengths of the internal byte slice iterator
        // to find length of current char
        let (pre_len, _) = s.iter.iter.size_hint();
        if let Some((i, c)) = s.next_back() {
            let (len, _) = s.iter.iter.size_hint();
            let char_len = pre_len - len;
            if self.char_eq.matches(c) {
                return SearchStep::Match(i, i + char_len);
            } else {
                return SearchStep::Reject(i, i + char_len);
            }
        }
        SearchStep::Done
    }
}

impl<'a, C: CharEq> DoubleEndedSearcher<'a> for CharEqSearcher<'a, C> {}

// Impl for &str

// TODO: Optimize the naive implementation here

#[derive(Clone)]
pub struct StrSearcher<'a, 'b> {
    haystack: &'a str,
    needle: &'b str,
    start: usize,
    end: usize,
    done: bool,
}

impl<'a, 'b> Pattern<'a> for &'b str {
    type Searcher = StrSearcher<'a, 'b>;

    #[inline]
    fn into_matcher(self, haystack: &'a str) -> StrSearcher<'a, 'b> {
        StrSearcher {
            haystack: haystack,
            needle: self,
            start: 0,
            end: haystack.len(),
            done: false,
        }
    }
}

unsafe impl<'a, 'b> Searcher<'a> for StrSearcher<'a, 'b>  {
    #[inline]
    fn haystack(&self) -> &'a str {
        self.haystack
    }

    #[inline]
    fn next(&mut self) -> SearchStep {
        str_search_step(self,
        |m: &mut StrSearcher| {
            // Forward step for empty needle
            let current_start = m.start;
            if !m.done {
                m.start = m.haystack.char_range_at(current_start).next;
            }
            SearchStep::Match(current_start, current_start)
        },
        |m: &mut StrSearcher| {
            // Forward step for nonempty needle
            let possible_match = &m.haystack[m.start .. m.start + m.needle.len()];
            let current_start = m.start;
            if possible_match == m.needle {
                m.start += m.needle.len();
                SearchStep::Match(current_start, m.start)
            } else {
                m.start += possible_match.chars().next().unwrap().len_utf8();
                SearchStep::Reject(current_start, m.start)
            }
        })
    }
}

unsafe impl<'a, 'b> ReverseSearcher<'a> for StrSearcher<'a, 'b>  {
    #[inline]
    fn next_back(&mut self) -> SearchStep {
        str_search_step(self,
        |m: &mut StrSearcher| {
            // Backward step for empty needle
            let current_end = m.end;
            if !m.done {
                m.end = m.haystack.char_range_at_reverse(current_end).next;
            }
            SearchStep::Match(current_end, current_end)
        },
        |m: &mut StrSearcher| {
            // Backward step for nonempty needle
            let possible_match = &m.haystack[m.end - m.needle.len() .. m.end];
            let current_end = m.end;
            if possible_match == m.needle {
                m.end -= m.needle.len();
                SearchStep::Match(m.end, current_end)
            } else {
                m.end -= possible_match.chars().rev().next().unwrap().len_utf8();
                SearchStep::Reject(m.end, current_end)
            }
        })
    }
}

fn str_search_step<F, G>(mut m: &mut StrSearcher, f: F, g: G) -> SearchStep
where F: FnOnce(&mut StrSearcher) -> SearchStep,
      G: FnOnce(&mut StrSearcher) -> SearchStep
{
    if m.done {
        SearchStep::Done
    } else if m.needle.len() == 0 && m.start <= m.end {
        // Case for needle == ""
        if m.start == m.end {
            m.done = true;
        }
        f(&mut m)
    } else if m.start + m.needle.len() <= m.end {
        // Case for needle != ""
        g(&mut m)
    } else {
        m.done = true;
        SearchStep::Done
    }
}
