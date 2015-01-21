// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::NamesIter::*;
pub use self::Regex::*;

use std::borrow::IntoCow;
use std::collections::HashMap;
use std::fmt;
use std::string::CowString;

use compile::Program;
use parse;
use vm;
use vm::{CaptureLocs, MatchKind, Exists, Location, Submatches};

/// Escapes all regular expression meta characters in `text`.
///
/// The string returned may be safely used as a literal in a regular
/// expression.
pub fn quote(text: &str) -> String {
    let mut quoted = String::with_capacity(text.len());
    for c in text.chars() {
        if parse::is_punct(c) {
            quoted.push('\\')
        }
        quoted.push(c);
    }
    quoted
}

/// Tests if the given regular expression matches somewhere in the text given.
///
/// If there was a problem compiling the regular expression, an error is
/// returned.
///
/// To find submatches, split or replace text, you'll need to compile an
/// expression first.
///
/// Note that you should prefer the `regex!` macro when possible. For example,
/// `regex!("...").is_match("...")`.
pub fn is_match(regex: &str, text: &str) -> Result<bool, parse::Error> {
    Regex::new(regex).map(|r| r.is_match(text))
}

/// A compiled regular expression
#[derive(Clone)]
pub enum Regex {
    // The representation of `Regex` is exported to support the `regex!`
    // syntax extension. Do not rely on it.
    //
    // See the comments for the `program` module in `lib.rs` for a more
    // detailed explanation for what `regex!` requires.
    #[doc(hidden)]
    Dynamic(ExDynamic),
    #[doc(hidden)]
    Native(ExNative),
}

#[derive(Clone)]
#[doc(hidden)]
pub struct ExDynamic {
    original: String,
    names: Vec<Option<String>>,
    #[doc(hidden)]
    pub prog: Program
}

#[doc(hidden)]
#[derive(Copy)]
pub struct ExNative {
    #[doc(hidden)]
    pub original: &'static str,
    #[doc(hidden)]
    pub names: &'static &'static [Option<&'static str>],
    #[doc(hidden)]
    pub prog: fn(MatchKind, &str, uint, uint) -> Vec<Option<uint>>
}

impl Clone for ExNative {
    fn clone(&self) -> ExNative {
        *self
    }
}

impl fmt::Display for Regex {
    /// Shows the original regular expression.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

impl Regex {
    /// Compiles a dynamic regular expression. Once compiled, it can be
    /// used repeatedly to search, split or replace text in a string.
    ///
    /// When possible, you should prefer the `regex!` macro since it is
    /// safer and always faster.
    ///
    /// If an invalid expression is given, then an error is returned.
    pub fn new(re: &str) -> Result<Regex, parse::Error> {
        let ast = try!(parse::parse(re));
        let (prog, names) = Program::new(ast);
        Ok(Dynamic(ExDynamic {
            original: re.to_string(),
            names: names,
            prog: prog,
        }))
    }

    /// Returns true if and only if the regex matches the string given.
    pub fn is_match(&self, text: &str) -> bool {
        has_match(&exec(self, Exists, text))
    }

    /// Returns the start and end byte range of the leftmost-first match in
    /// `text`. If no match exists, then `None` is returned.
    pub fn find(&self, text: &str) -> Option<(uint, uint)> {
        let caps = exec(self, Location, text);
        if has_match(&caps) {
            Some((caps[0].unwrap(), caps[1].unwrap()))
        } else {
            None
        }
    }

    /// Returns an iterator for each successive non-overlapping match in
    /// `text`, returning the start and end byte indices with respect to
    /// `text`.
    pub fn find_iter<'r, 't>(&'r self, text: &'t str) -> FindMatches<'r, 't> {
        FindMatches {
            re: self,
            search: text,
            last_end: 0,
            last_match: None,
        }
    }

    /// Returns the capture groups corresponding to the leftmost-first
    /// match in `text`. Capture group `0` always corresponds to the entire
    /// match. If no match is found, then `None` is returned.
    ///
    /// You should only use `captures` if you need access to submatches.
    /// Otherwise, `find` is faster for discovering the location of the overall
    /// match.
    pub fn captures<'t>(&self, text: &'t str) -> Option<Captures<'t>> {
        let caps = exec(self, Submatches, text);
        Captures::new(self, text, caps)
    }

    /// Returns an iterator over all the non-overlapping capture groups matched
    /// in `text`. This is operationally the same as `find_iter` (except it
    /// yields information about submatches).
    pub fn captures_iter<'r, 't>(&'r self, text: &'t str)
                                -> FindCaptures<'r, 't> {
        FindCaptures {
            re: self,
            search: text,
            last_match: None,
            last_end: 0,
        }
    }

    /// Returns an iterator of substrings of `text` delimited by a match
    /// of the regular expression.
    /// Namely, each element of the iterator corresponds to text that *isn't*
    /// matched by the regular expression.
    ///
    /// This method will *not* copy the text given.
    pub fn split<'r, 't>(&'r self, text: &'t str) -> RegexSplits<'r, 't> {
        RegexSplits {
            finder: self.find_iter(text),
            last: 0,
        }
    }

    /// Returns an iterator of at most `limit` substrings of `text` delimited
    /// by a match of the regular expression. (A `limit` of `0` will return no
    /// substrings.)
    /// Namely, each element of the iterator corresponds to text that *isn't*
    /// matched by the regular expression.
    /// The remainder of the string that is not split will be the last element
    /// in the iterator.
    ///
    /// This method will *not* copy the text given.
    pub fn splitn<'r, 't>(&'r self, text: &'t str, limit: uint)
                         -> RegexSplitsN<'r, 't> {
        RegexSplitsN {
            splits: self.split(text),
            cur: 0,
            limit: limit,
        }
    }

    /// Replaces the leftmost-first match with the replacement provided.
    /// The replacement can be a regular string (where `$N` and `$name` are
    /// expanded to match capture groups) or a function that takes the matches'
    /// `Captures` and returns the replaced string.
    ///
    /// If no match is found, then a copy of the string is returned unchanged.
    pub fn replace<R: Replacer>(&self, text: &str, rep: R) -> String {
        self.replacen(text, 1, rep)
    }

    /// Replaces all non-overlapping matches in `text` with the
    /// replacement provided. This is the same as calling `replacen` with
    /// `limit` set to `0`.
    ///
    /// See the documentation for `replace` for details on how to access
    /// submatches in the replacement string.
    pub fn replace_all<R: Replacer>(&self, text: &str, rep: R) -> String {
        self.replacen(text, 0, rep)
    }

    /// Replaces at most `limit` non-overlapping matches in `text` with the
    /// replacement provided. If `limit` is 0, then all non-overlapping matches
    /// are replaced.
    ///
    /// See the documentation for `replace` for details on how to access
    /// submatches in the replacement string.
    pub fn replacen<R: Replacer>
                   (&self, text: &str, limit: uint, mut rep: R) -> String {
        let mut new = String::with_capacity(text.len());
        let mut last_match = 0u;

        for (i, cap) in self.captures_iter(text).enumerate() {
            // It'd be nicer to use the 'take' iterator instead, but it seemed
            // awkward given that '0' => no limit.
            if limit > 0 && i >= limit {
                break
            }

            let (s, e) = cap.pos(0).unwrap(); // captures only reports matches
            new.push_str(&text[last_match..s]);
            new.push_str(&rep.reg_replace(&cap)[]);
            last_match = e;
        }
        new.push_str(&text[last_match..text.len()]);
        return new;
    }

    /// Returns the original string of this regex.
    pub fn as_str<'a>(&'a self) -> &'a str {
        match *self {
            Dynamic(ExDynamic { ref original, .. }) => &original[],
            Native(ExNative { ref original, .. }) => &original[],
        }
    }

    #[doc(hidden)]
    #[unstable]
    pub fn names_iter<'a>(&'a self) -> NamesIter<'a> {
        match *self {
            Native(ref n) => NamesIterNative(n.names.iter()),
            Dynamic(ref d) => NamesIterDynamic(d.names.iter())
        }
    }

    fn names_len(&self) -> uint {
        match *self {
            Native(ref n) => n.names.len(),
            Dynamic(ref d) => d.names.len()
        }
    }

}

#[derive(Clone)]
pub enum NamesIter<'a> {
    NamesIterNative(::std::slice::Iter<'a, Option<&'static str>>),
    NamesIterDynamic(::std::slice::Iter<'a, Option<String>>)
}

impl<'a> Iterator for NamesIter<'a> {
    type Item = Option<String>;

    fn next(&mut self) -> Option<Option<String>> {
        match *self {
            NamesIterNative(ref mut i) => i.next().map(|x| x.map(|s| s.to_string())),
            NamesIterDynamic(ref mut i) => i.next().map(|x| x.as_ref().map(|s| s.to_string())),
        }
    }
}

/// NoExpand indicates literal string replacement.
///
/// It can be used with `replace` and `replace_all` to do a literal
/// string replacement without expanding `$name` to their corresponding
/// capture groups.
///
/// `'r` is the lifetime of the literal text.
pub struct NoExpand<'t>(pub &'t str);

/// Replacer describes types that can be used to replace matches in a string.
pub trait Replacer {
    /// Returns a possibly owned string that is used to replace the match
    /// corresponding to the `caps` capture group.
    ///
    /// The `'a` lifetime refers to the lifetime of a borrowed string when
    /// a new owned string isn't needed (e.g., for `NoExpand`).
    fn reg_replace<'a>(&'a mut self, caps: &Captures) -> CowString<'a>;
}

impl<'t> Replacer for NoExpand<'t> {
    fn reg_replace<'a>(&'a mut self, _: &Captures) -> CowString<'a> {
        let NoExpand(s) = *self;
        s.into_cow()
    }
}

impl<'t> Replacer for &'t str {
    fn reg_replace<'a>(&'a mut self, caps: &Captures) -> CowString<'a> {
        caps.expand(*self).into_cow()
    }
}

impl<F> Replacer for F where F: FnMut(&Captures) -> String {
    fn reg_replace<'a>(&'a mut self, caps: &Captures) -> CowString<'a> {
        (*self)(caps).into_cow()
    }
}

/// Yields all substrings delimited by a regular expression match.
///
/// `'r` is the lifetime of the compiled expression and `'t` is the lifetime
/// of the string being split.
#[derive(Clone)]
pub struct RegexSplits<'r, 't> {
    finder: FindMatches<'r, 't>,
    last: uint,
}

impl<'r, 't> Iterator for RegexSplits<'r, 't> {
    type Item = &'t str;

    fn next(&mut self) -> Option<&'t str> {
        let text = self.finder.search;
        match self.finder.next() {
            None => {
                if self.last >= text.len() {
                    None
                } else {
                    let s = &text[self.last..text.len()];
                    self.last = text.len();
                    Some(s)
                }
            }
            Some((s, e)) => {
                let matched = &text[self.last..s];
                self.last = e;
                Some(matched)
            }
        }
    }
}

/// Yields at most `N` substrings delimited by a regular expression match.
///
/// The last substring will be whatever remains after splitting.
///
/// `'r` is the lifetime of the compiled expression and `'t` is the lifetime
/// of the string being split.
#[derive(Clone)]
pub struct RegexSplitsN<'r, 't> {
    splits: RegexSplits<'r, 't>,
    cur: uint,
    limit: uint,
}

impl<'r, 't> Iterator for RegexSplitsN<'r, 't> {
    type Item = &'t str;

    fn next(&mut self) -> Option<&'t str> {
        let text = self.splits.finder.search;
        if self.cur >= self.limit {
            None
        } else {
            self.cur += 1;
            if self.cur >= self.limit {
                Some(&text[self.splits.last..text.len()])
            } else {
                self.splits.next()
            }
        }
    }
}

/// Captures represents a group of captured strings for a single match.
///
/// The 0th capture always corresponds to the entire match. Each subsequent
/// index corresponds to the next capture group in the regex.
/// If a capture group is named, then the matched string is *also* available
/// via the `name` method. (Note that the 0th capture is always unnamed and so
/// must be accessed with the `at` method.)
///
/// Positions returned from a capture group are always byte indices.
///
/// `'t` is the lifetime of the matched text.
pub struct Captures<'t> {
    text: &'t str,
    locs: CaptureLocs,
    named: Option<HashMap<String, uint>>,
}

impl<'t> Captures<'t> {
    #[allow(unstable)]
    fn new(re: &Regex, search: &'t str, locs: CaptureLocs)
          -> Option<Captures<'t>> {
        if !has_match(&locs) {
            return None
        }

        let named =
            if re.names_len() == 0 {
                None
            } else {
                let mut named = HashMap::new();
                for (i, name) in re.names_iter().enumerate() {
                    match name {
                        None => {},
                        Some(name) => {
                            named.insert(name, i);
                        }
                    }
                }
                Some(named)
            };
        Some(Captures {
            text: search,
            locs: locs,
            named: named,
        })
    }

    /// Returns the start and end positions of the Nth capture group.
    /// Returns `None` if `i` is not a valid capture group or if the capture
    /// group did not match anything.
    /// The positions returned are *always* byte indices with respect to the
    /// original string matched.
    pub fn pos(&self, i: uint) -> Option<(uint, uint)> {
        let (s, e) = (i * 2, i * 2 + 1);
        if e >= self.locs.len() || self.locs[s].is_none() {
            // VM guarantees that each pair of locations are both Some or None.
            return None
        }
        Some((self.locs[s].unwrap(), self.locs[e].unwrap()))
    }

    /// Returns the matched string for the capture group `i`.  If `i` isn't
    /// a valid capture group or didn't match anything, then `None` is
    /// returned.
    pub fn at(&self, i: uint) -> Option<&'t str> {
        match self.pos(i) {
            None => None,
            Some((s, e)) => Some(&self.text[s.. e])
        }
    }

    /// Returns the matched string for the capture group named `name`.  If
    /// `name` isn't a valid capture group or didn't match anything, then
    /// `None` is returned.
    pub fn name(&self, name: &str) -> Option<&'t str> {
        match self.named {
            None => None,
            Some(ref h) => {
                match h.get(name) {
                    None => None,
                    Some(i) => self.at(*i),
                }
            }
        }
    }

    /// Creates an iterator of all the capture groups in order of appearance
    /// in the regular expression.
    pub fn iter(&'t self) -> SubCaptures<'t> {
        SubCaptures { idx: 0, caps: self, }
    }

    /// Creates an iterator of all the capture group positions in order of
    /// appearance in the regular expression. Positions are byte indices
    /// in terms of the original string matched.
    pub fn iter_pos(&'t self) -> SubCapturesPos<'t> {
        SubCapturesPos { idx: 0, caps: self, }
    }

    /// Expands all instances of `$name` in `text` to the corresponding capture
    /// group `name`.
    ///
    /// `name` may be an integer corresponding to the index of the
    /// capture group (counted by order of opening parenthesis where `0` is the
    /// entire match) or it can be a name (consisting of letters, digits or
    /// underscores) corresponding to a named capture group.
    ///
    /// If `name` isn't a valid capture group (whether the name doesn't exist or
    /// isn't a valid index), then it is replaced with the empty string.
    ///
    /// To write a literal `$` use `$$`.
    pub fn expand(&self, text: &str) -> String {
        // How evil can you get?
        // FIXME: Don't use regexes for this. It's completely unnecessary.
        let re = Regex::new(r"(^|[^$]|\b)\$(\w+)").unwrap();
        let text = re.replace_all(text, |&mut: refs: &Captures| -> String {
            let pre = refs.at(1).unwrap_or("");
            let name = refs.at(2).unwrap_or("");
            format!("{}{}", pre,
                    match name.parse::<uint>() {
                None => self.name(name).unwrap_or("").to_string(),
                Some(i) => self.at(i).unwrap_or("").to_string(),
            })
        });
        let re = Regex::new(r"\$\$").unwrap();
        re.replace_all(&text[], NoExpand("$"))
    }

    /// Returns the number of captured groups.
    #[inline]
    pub fn len(&self) -> uint { self.locs.len() / 2 }

    /// Returns if there are no captured groups.
    #[inline]
    pub fn is_empty(&self) -> bool { self.len() == 0 }
}

/// An iterator over capture groups for a particular match of a regular
/// expression.
///
/// `'t` is the lifetime of the matched text.
#[derive(Clone)]
pub struct SubCaptures<'t> {
    idx: uint,
    caps: &'t Captures<'t>,
}

impl<'t> Iterator for SubCaptures<'t> {
    type Item = &'t str;

    fn next(&mut self) -> Option<&'t str> {
        if self.idx < self.caps.len() {
            self.idx += 1;
            Some(self.caps.at(self.idx - 1).unwrap_or(""))
        } else {
            None
        }
    }
}

/// An iterator over capture group positions for a particular match of a
/// regular expression.
///
/// Positions are byte indices in terms of the original string matched.
///
/// `'t` is the lifetime of the matched text.
#[derive(Clone)]
pub struct SubCapturesPos<'t> {
    idx: uint,
    caps: &'t Captures<'t>,
}

impl<'t> Iterator for SubCapturesPos<'t> {
    type Item = Option<(uint, uint)>;

    fn next(&mut self) -> Option<Option<(uint, uint)>> {
        if self.idx < self.caps.len() {
            self.idx += 1;
            Some(self.caps.pos(self.idx - 1))
        } else {
            None
        }
    }
}

/// An iterator that yields all non-overlapping capture groups matching a
/// particular regular expression.
///
/// The iterator stops when no more matches can be found.
///
/// `'r` is the lifetime of the compiled expression and `'t` is the lifetime
/// of the matched string.
#[derive(Clone)]
pub struct FindCaptures<'r, 't> {
    re: &'r Regex,
    search: &'t str,
    last_match: Option<uint>,
    last_end: uint,
}

impl<'r, 't> Iterator for FindCaptures<'r, 't> {
    type Item = Captures<'t>;

    fn next(&mut self) -> Option<Captures<'t>> {
        if self.last_end > self.search.len() {
            return None
        }

        let caps = exec_slice(self.re, Submatches, self.search,
                              self.last_end, self.search.len());
        let (s, e) =
            if !has_match(&caps) {
                return None
            } else {
                (caps[0].unwrap(), caps[1].unwrap())
            };

        // Don't accept empty matches immediately following a match.
        // i.e., no infinite loops please.
        if e == s && Some(self.last_end) == self.last_match {
            self.last_end += 1;
            return self.next()
        }
        self.last_end = e;
        self.last_match = Some(self.last_end);
        Captures::new(self.re, self.search, caps)
    }
}

/// An iterator over all non-overlapping matches for a particular string.
///
/// The iterator yields a tuple of integers corresponding to the start and end
/// of the match. The indices are byte offsets. The iterator stops when no more
/// matches can be found.
///
/// `'r` is the lifetime of the compiled expression and `'t` is the lifetime
/// of the matched string.
#[derive(Clone)]
pub struct FindMatches<'r, 't> {
    re: &'r Regex,
    search: &'t str,
    last_match: Option<uint>,
    last_end: uint,
}

impl<'r, 't> Iterator for FindMatches<'r, 't> {
    type Item = (uint, uint);

    fn next(&mut self) -> Option<(uint, uint)> {
        if self.last_end > self.search.len() {
            return None
        }

        let caps = exec_slice(self.re, Location, self.search,
                              self.last_end, self.search.len());
        let (s, e) =
            if !has_match(&caps) {
                return None
            } else {
                (caps[0].unwrap(), caps[1].unwrap())
            };

        // Don't accept empty matches immediately following a match.
        // i.e., no infinite loops please.
        if e == s && Some(self.last_end) == self.last_match {
            self.last_end += 1;
            return self.next()
        }
        self.last_end = e;
        self.last_match = Some(self.last_end);
        Some((s, e))
    }
}

fn exec(re: &Regex, which: MatchKind, input: &str) -> CaptureLocs {
    exec_slice(re, which, input, 0, input.len())
}

fn exec_slice(re: &Regex, which: MatchKind,
              input: &str, s: uint, e: uint) -> CaptureLocs {
    match *re {
        Dynamic(ExDynamic { ref prog, .. }) => vm::run(which, prog, input, s, e),
        Native(ExNative { ref prog, .. }) => (*prog)(which, input, s, e),
    }
}

#[inline]
fn has_match(caps: &CaptureLocs) -> bool {
    caps.len() >= 2 && caps[0].is_some() && caps[1].is_some()
}
