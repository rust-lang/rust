use std::mem;

use aho_corasick::{self, packed, AhoCorasick};
use memchr::{memchr, memchr2, memchr3, memmem};
use regex_syntax::hir::literal::{Literal, Seq};

/// A prefix extracted from a compiled regular expression.
///
/// A regex prefix is a set of literal strings that *must* be matched at the
/// beginning of a regex in order for the entire regex to match. Similarly
/// for a regex suffix.
#[derive(Clone, Debug)]
pub struct LiteralSearcher {
    complete: bool,
    lcp: Memmem,
    lcs: Memmem,
    matcher: Matcher,
}

#[derive(Clone, Debug)]
enum Matcher {
    /// No literals. (Never advances through the input.)
    Empty,
    /// A set of four or more single byte literals.
    Bytes(SingleByteSet),
    /// A single substring, using vector accelerated routines when available.
    Memmem(Memmem),
    /// An Aho-Corasick automaton.
    AC { ac: AhoCorasick, lits: Vec<Literal> },
    /// A packed multiple substring searcher, using SIMD.
    ///
    /// Note that Aho-Corasick will actually use this packed searcher
    /// internally automatically, however, there is some overhead associated
    /// with going through the Aho-Corasick machinery. So using the packed
    /// searcher directly results in some gains.
    Packed { s: packed::Searcher, lits: Vec<Literal> },
}

impl LiteralSearcher {
    /// Returns a matcher that never matches and never advances the input.
    pub fn empty() -> Self {
        Self::new(Seq::infinite(), Matcher::Empty)
    }

    /// Returns a matcher for literal prefixes from the given set.
    pub fn prefixes(lits: Seq) -> Self {
        let matcher = Matcher::prefixes(&lits);
        Self::new(lits, matcher)
    }

    /// Returns a matcher for literal suffixes from the given set.
    pub fn suffixes(lits: Seq) -> Self {
        let matcher = Matcher::suffixes(&lits);
        Self::new(lits, matcher)
    }

    fn new(lits: Seq, matcher: Matcher) -> Self {
        LiteralSearcher {
            complete: lits.is_exact(),
            lcp: Memmem::new(lits.longest_common_prefix().unwrap_or(b"")),
            lcs: Memmem::new(lits.longest_common_suffix().unwrap_or(b"")),
            matcher,
        }
    }

    /// Returns true if all matches comprise the entire regular expression.
    ///
    /// This does not necessarily mean that a literal match implies a match
    /// of the regular expression. For example, the regular expression `^a`
    /// is comprised of a single complete literal `a`, but the regular
    /// expression demands that it only match at the beginning of a string.
    pub fn complete(&self) -> bool {
        self.complete && !self.is_empty()
    }

    /// Find the position of a literal in `haystack` if it exists.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub fn find(&self, haystack: &[u8]) -> Option<(usize, usize)> {
        use self::Matcher::*;
        match self.matcher {
            Empty => Some((0, 0)),
            Bytes(ref sset) => sset.find(haystack).map(|i| (i, i + 1)),
            Memmem(ref s) => s.find(haystack).map(|i| (i, i + s.len())),
            AC { ref ac, .. } => {
                ac.find(haystack).map(|m| (m.start(), m.end()))
            }
            Packed { ref s, .. } => {
                s.find(haystack).map(|m| (m.start(), m.end()))
            }
        }
    }

    /// Like find, except matches must start at index `0`.
    pub fn find_start(&self, haystack: &[u8]) -> Option<(usize, usize)> {
        for lit in self.iter() {
            if lit.len() > haystack.len() {
                continue;
            }
            if lit == &haystack[0..lit.len()] {
                return Some((0, lit.len()));
            }
        }
        None
    }

    /// Like find, except matches must end at index `haystack.len()`.
    pub fn find_end(&self, haystack: &[u8]) -> Option<(usize, usize)> {
        for lit in self.iter() {
            if lit.len() > haystack.len() {
                continue;
            }
            if lit == &haystack[haystack.len() - lit.len()..] {
                return Some((haystack.len() - lit.len(), haystack.len()));
            }
        }
        None
    }

    /// Returns an iterator over all literals to be matched.
    pub fn iter(&self) -> LiteralIter<'_> {
        match self.matcher {
            Matcher::Empty => LiteralIter::Empty,
            Matcher::Bytes(ref sset) => LiteralIter::Bytes(&sset.dense),
            Matcher::Memmem(ref s) => LiteralIter::Single(&s.finder.needle()),
            Matcher::AC { ref lits, .. } => LiteralIter::AC(lits),
            Matcher::Packed { ref lits, .. } => LiteralIter::Packed(lits),
        }
    }

    /// Returns a matcher for the longest common prefix of this matcher.
    pub fn lcp(&self) -> &Memmem {
        &self.lcp
    }

    /// Returns a matcher for the longest common suffix of this matcher.
    pub fn lcs(&self) -> &Memmem {
        &self.lcs
    }

    /// Returns true iff this prefix is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of prefixes in this machine.
    pub fn len(&self) -> usize {
        use self::Matcher::*;
        match self.matcher {
            Empty => 0,
            Bytes(ref sset) => sset.dense.len(),
            Memmem(_) => 1,
            AC { ref ac, .. } => ac.patterns_len(),
            Packed { ref lits, .. } => lits.len(),
        }
    }

    /// Return the approximate heap usage of literals in bytes.
    pub fn approximate_size(&self) -> usize {
        use self::Matcher::*;
        match self.matcher {
            Empty => 0,
            Bytes(ref sset) => sset.approximate_size(),
            Memmem(ref single) => single.approximate_size(),
            AC { ref ac, .. } => ac.memory_usage(),
            Packed { ref s, .. } => s.memory_usage(),
        }
    }
}

impl Matcher {
    fn prefixes(lits: &Seq) -> Self {
        let sset = SingleByteSet::prefixes(lits);
        Matcher::new(lits, sset)
    }

    fn suffixes(lits: &Seq) -> Self {
        let sset = SingleByteSet::suffixes(lits);
        Matcher::new(lits, sset)
    }

    fn new(lits: &Seq, sset: SingleByteSet) -> Self {
        if lits.is_empty() || lits.min_literal_len() == Some(0) {
            return Matcher::Empty;
        }
        let lits = match lits.literals() {
            None => return Matcher::Empty,
            Some(members) => members,
        };
        if sset.dense.len() >= 26 {
            // Avoid trying to match a large number of single bytes.
            // This is *very* sensitive to a frequency analysis comparison
            // between the bytes in sset and the composition of the haystack.
            // No matter the size of sset, if its members all are rare in the
            // haystack, then it'd be worth using it. How to tune this... IDK.
            // ---AG
            return Matcher::Empty;
        }
        if sset.complete {
            return Matcher::Bytes(sset);
        }
        if lits.len() == 1 {
            return Matcher::Memmem(Memmem::new(lits[0].as_bytes()));
        }

        let pats: Vec<&[u8]> = lits.iter().map(|lit| lit.as_bytes()).collect();
        let is_aho_corasick_fast = sset.dense.len() <= 1 && sset.all_ascii;
        if lits.len() <= 100 && !is_aho_corasick_fast {
            let mut builder = packed::Config::new()
                .match_kind(packed::MatchKind::LeftmostFirst)
                .builder();
            if let Some(s) = builder.extend(&pats).build() {
                return Matcher::Packed { s, lits: lits.to_owned() };
            }
        }
        let ac = AhoCorasick::builder()
            .match_kind(aho_corasick::MatchKind::LeftmostFirst)
            .kind(Some(aho_corasick::AhoCorasickKind::DFA))
            .build(&pats)
            .unwrap();
        Matcher::AC { ac, lits: lits.to_owned() }
    }
}

#[derive(Debug)]
pub enum LiteralIter<'a> {
    Empty,
    Bytes(&'a [u8]),
    Single(&'a [u8]),
    AC(&'a [Literal]),
    Packed(&'a [Literal]),
}

impl<'a> Iterator for LiteralIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        match *self {
            LiteralIter::Empty => None,
            LiteralIter::Bytes(ref mut many) => {
                if many.is_empty() {
                    None
                } else {
                    let next = &many[0..1];
                    *many = &many[1..];
                    Some(next)
                }
            }
            LiteralIter::Single(ref mut one) => {
                if one.is_empty() {
                    None
                } else {
                    let next = &one[..];
                    *one = &[];
                    Some(next)
                }
            }
            LiteralIter::AC(ref mut lits) => {
                if lits.is_empty() {
                    None
                } else {
                    let next = &lits[0];
                    *lits = &lits[1..];
                    Some(next.as_bytes())
                }
            }
            LiteralIter::Packed(ref mut lits) => {
                if lits.is_empty() {
                    None
                } else {
                    let next = &lits[0];
                    *lits = &lits[1..];
                    Some(next.as_bytes())
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
struct SingleByteSet {
    sparse: Vec<bool>,
    dense: Vec<u8>,
    complete: bool,
    all_ascii: bool,
}

impl SingleByteSet {
    fn new() -> SingleByteSet {
        SingleByteSet {
            sparse: vec![false; 256],
            dense: vec![],
            complete: true,
            all_ascii: true,
        }
    }

    fn prefixes(lits: &Seq) -> SingleByteSet {
        let mut sset = SingleByteSet::new();
        let lits = match lits.literals() {
            None => return sset,
            Some(lits) => lits,
        };
        for lit in lits.iter() {
            sset.complete = sset.complete && lit.len() == 1;
            if let Some(&b) = lit.as_bytes().get(0) {
                if !sset.sparse[b as usize] {
                    if b > 0x7F {
                        sset.all_ascii = false;
                    }
                    sset.dense.push(b);
                    sset.sparse[b as usize] = true;
                }
            }
        }
        sset
    }

    fn suffixes(lits: &Seq) -> SingleByteSet {
        let mut sset = SingleByteSet::new();
        let lits = match lits.literals() {
            None => return sset,
            Some(lits) => lits,
        };
        for lit in lits.iter() {
            sset.complete = sset.complete && lit.len() == 1;
            if let Some(&b) = lit.as_bytes().last() {
                if !sset.sparse[b as usize] {
                    if b > 0x7F {
                        sset.all_ascii = false;
                    }
                    sset.dense.push(b);
                    sset.sparse[b as usize] = true;
                }
            }
        }
        sset
    }

    /// Faster find that special cases certain sizes to use memchr.
    #[cfg_attr(feature = "perf-inline", inline(always))]
    fn find(&self, text: &[u8]) -> Option<usize> {
        match self.dense.len() {
            0 => None,
            1 => memchr(self.dense[0], text),
            2 => memchr2(self.dense[0], self.dense[1], text),
            3 => memchr3(self.dense[0], self.dense[1], self.dense[2], text),
            _ => self._find(text),
        }
    }

    /// Generic find that works on any sized set.
    fn _find(&self, haystack: &[u8]) -> Option<usize> {
        for (i, &b) in haystack.iter().enumerate() {
            if self.sparse[b as usize] {
                return Some(i);
            }
        }
        None
    }

    fn approximate_size(&self) -> usize {
        (self.dense.len() * mem::size_of::<u8>())
            + (self.sparse.len() * mem::size_of::<bool>())
    }
}

/// A simple wrapper around the memchr crate's memmem implementation.
///
/// The API this exposes mirrors the API of previous substring searchers that
/// this supplanted.
#[derive(Clone, Debug)]
pub struct Memmem {
    finder: memmem::Finder<'static>,
    char_len: usize,
}

impl Memmem {
    fn new(pat: &[u8]) -> Memmem {
        Memmem {
            finder: memmem::Finder::new(pat).into_owned(),
            char_len: char_len_lossy(pat),
        }
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub fn find(&self, haystack: &[u8]) -> Option<usize> {
        self.finder.find(haystack)
    }

    #[cfg_attr(feature = "perf-inline", inline(always))]
    pub fn is_suffix(&self, text: &[u8]) -> bool {
        if text.len() < self.len() {
            return false;
        }
        &text[text.len() - self.len()..] == self.finder.needle()
    }

    pub fn len(&self) -> usize {
        self.finder.needle().len()
    }

    pub fn char_len(&self) -> usize {
        self.char_len
    }

    fn approximate_size(&self) -> usize {
        self.finder.needle().len() * mem::size_of::<u8>()
    }
}

fn char_len_lossy(bytes: &[u8]) -> usize {
    String::from_utf8_lossy(bytes).chars().count()
}
