// run-pass
#![allow(unused_must_use)]
#![allow(dead_code)]
// pretty-expanded FIXME #23616

trait Matcher {
    fn next_match(&mut self) -> Option<(usize, usize)>;
}

struct CharPredMatcher<'a, 'b> {
    str: &'a str,
    pred: Box<dyn FnMut(char) -> bool + 'b>,
}

impl<'a, 'b> Matcher for CharPredMatcher<'a, 'b> {
    fn next_match(&mut self) -> Option<(usize, usize)> {
        None
    }
}

trait IntoMatcher<'a, T> {
    fn into_matcher(self, _: &'a str) -> T;
}

impl<'a, 'b, F> IntoMatcher<'a, CharPredMatcher<'a, 'b>> for F where F: FnMut(char) -> bool + 'b {
    fn into_matcher(self, s: &'a str) -> CharPredMatcher<'a, 'b> {
        CharPredMatcher {
            str: s,
            pred: Box::new(self),
        }
    }
}

struct MatchIndices<M> {
    matcher: M
}

impl<M: Matcher> Iterator for MatchIndices<M> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
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
        .collect::<Vec<_>>();
}
