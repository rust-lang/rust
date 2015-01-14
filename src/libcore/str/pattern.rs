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
    type Matcher: Matcher<'a>;
    fn into_matcher(self, haystack: &'a str) -> Self::Matcher;

    #[inline]
    fn is_contained_in(self, haystack: &'a str) -> bool {
        Matcher::next(&mut self.into_matcher(haystack)).is_some()
    }
}

// Matcher

pub unsafe trait Matcher<'a> {
    fn haystack(&self) -> &'a str;
    fn next(&mut self) -> Option<(usize, usize)>;
}

pub unsafe trait ReverseMatcher<'a>: Matcher<'a> {
    fn next_back(&mut self) -> Option<(usize, usize)>;
}

pub trait DoubleEndedMatcher<'a>: ReverseMatcher<'a> {}

// Impl for CharEq

struct CharEqMatcher<'a, C>(C, &'a str, super::CharIndices<'a>);

impl<'a, C: CharEq> Pattern<'a> for C {
    type Matcher = CharEqMatcher<'a, C>;

    #[inline]
    fn into_matcher(self, haystack: &'a str) -> CharEqMatcher<'a, C> {
        CharEqMatcher(self, haystack, haystack.char_indices())
    }
}

unsafe impl<'a, C: CharEq> Matcher<'a> for CharEqMatcher<'a, C> {
    #[inline]
    fn haystack(&self) -> &'a str {
        self.1
    }

    #[inline]
    fn next(&mut self) -> Option<(usize, usize)> {
        while let Some((i, c)) = self.2.next() {
            if self.0.matches(c) {
                return Some((i, i + c.len_utf8()));
            }
        }
        None
    }
}

unsafe impl<'a, C: CharEq> ReverseMatcher<'a> for CharEqMatcher<'a, C> {
    #[inline]
    fn next_back(&mut self) -> Option<(usize, usize)> {
        while let Some((i, c)) = self.2.next_back() {
            if self.0.matches(c) {
                return Some((i, i + c.len_utf8()));
            }
        }
        None
    }
}

impl<'a, C: CharEq> DoubleEndedMatcher<'a> for CharEqMatcher<'a, C> {}

// Impl for &str

struct StrMatcher<'a>(super::OldMatchIndices<'a>);

impl<'a> Pattern<'a> for &'a str {
    type Matcher = StrMatcher<'a>;

    #[inline]
    fn into_matcher(self, haystack: &'a str) -> StrMatcher<'a> {
        let mi = super::OldMatchIndices {
            haystack: haystack,
            needle: self,
            searcher: super::Searcher::new(haystack.as_bytes(), self.as_bytes())
        };
        StrMatcher(mi)
    }
}

unsafe impl<'a> Matcher<'a> for StrMatcher<'a>  {
    #[inline]
    fn haystack(&self) -> &'a str {
        self.0.haystack
    }

    #[inline]
    fn next(&mut self) -> Option<(usize, usize)> {
        self.0.next()
    }
}
