// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Matcher {
    fn next_match(&mut self) -> Option<(uint, uint)>;
}

struct CharPredMatcher<'a, 'b> {
    str: &'a str,
    pred: |char|:'b -> bool
}

impl<'a, 'b> Matcher for CharPredMatcher<'a, 'b> {
    fn next_match(&mut self) -> Option<(uint, uint)> {
        None
    }
}

trait IntoMatcher<'a, T> {
    fn into_matcher(self, &'a str) -> T;
}

impl<'a, 'b> IntoMatcher<'a, CharPredMatcher<'a, 'b>> for |char|:'b -> bool {
    fn into_matcher(self, s: &'a str) -> CharPredMatcher<'a, 'b> {
        CharPredMatcher {
            str: s,
            pred: self
        }
    }
}

struct MatchIndices<M> {
    matcher: M
}

impl<M: Matcher> Iterator<(uint, uint)> for MatchIndices<M> {
    fn next(&mut self) -> Option<(uint, uint)> {
        self.matcher.next_match()
    }
}

fn match_indices<'a, M, T: IntoMatcher<'a, M>>(s: &'a str, from: T) -> MatchIndices<M> {
    let string_matcher = from.into_matcher(s);
    MatchIndices { matcher: string_matcher }
}

fn main() {
    let s = "abcbdef";
    match_indices(s, |c: char| c == 'b')
        .collect::<Vec<(uint, uint)>>();
}
