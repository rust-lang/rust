// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: Currently, the VM simulates an NFA. It would be nice to have another
// VM that simulates a DFA.
//
// According to Russ Cox[1], a DFA performs better than an NFA, principally
// because it reuses states previously computed by the machine *and* doesn't
// keep track of capture groups. The drawback of a DFA (aside from its
// complexity) is that it can't accurately return the locations of submatches.
// The NFA *can* do that. (This is my understanding anyway.)
//
// Cox suggests that a DFA ought to be used to answer "does this match" and
// "where does it match" questions. (In the latter, the starting position of
// the match is computed by executing the regex backwards.) Cox also suggests
// that a DFA should be run when asking "where are the submatches", which can
// 1) quickly answer "no" is there's no match and 2) discover the substring
// that matches, which means running the NFA on smaller input.
//
// Currently, the NFA simulation implemented below does some dirty tricks to
// avoid tracking capture groups when they aren't needed (which only works
// for 'is_match', not 'find'). This is a half-measure, but does provide some
// perf improvement.
//
// AFAIK, the DFA/NFA approach is implemented in RE2/C++ but *not* in RE2/Go.
//
// [1] - http://swtch.com/~rsc/regex/regex3.html

use std::cmp;
use std::mem;
use std::slice::MutableSlice;
use compile::{
    Program,
    Match, OneChar, CharClass, Any, EmptyBegin, EmptyEnd, EmptyWordBoundary,
    Save, Jump, Split,
};
use parse::{FLAG_NOCASE, FLAG_MULTI, FLAG_DOTNL, FLAG_NEGATED};
use unicode::regex::PERLW;

pub type CaptureLocs = Vec<Option<uint>>;

/// Indicates the type of match to be performed by the VM.
pub enum MatchKind {
    /// Only checks if a match exists or not. Does not return location.
    Exists,
    /// Returns the start and end indices of the entire match in the input
    /// given.
    Location,
    /// Returns the start and end indices of each submatch in the input given.
    Submatches,
}

/// Runs an NFA simulation on the compiled expression given on the search text
/// `input`. The search begins at byte index `start` and ends at byte index
/// `end`. (The range is specified here so that zero-width assertions will work
/// correctly when searching for successive non-overlapping matches.)
///
/// The `which` parameter indicates what kind of capture information the caller
/// wants. There are three choices: match existence only, the location of the
/// entire match or the locations of the entire match in addition to the
/// locations of each submatch.
pub fn run<'r, 't>(which: MatchKind, prog: &'r Program, input: &'t str,
                   start: uint, end: uint) -> CaptureLocs {
    Nfa {
        which: which,
        prog: prog,
        input: input,
        start: start,
        end: end,
        ic: 0,
        chars: CharReader::new(input),
    }.run()
}

struct Nfa<'r, 't> {
    which: MatchKind,
    prog: &'r Program,
    input: &'t str,
    start: uint,
    end: uint,
    ic: uint,
    chars: CharReader<'t>,
}

/// Indicates the next action to take after a single non-empty instruction
/// is processed.
pub enum StepState {
    /// This is returned if and only if a Match instruction is reached and
    /// we only care about the existence of a match. It instructs the VM to
    /// quit early.
    StepMatchEarlyReturn,
    /// Indicates that a match was found. Thus, the rest of the states in the
    /// *current* queue should be dropped (i.e., leftmost-first semantics).
    /// States in the "next" queue can still be processed.
    StepMatch,
    /// No match was found. Continue with the next state in the queue.
    StepContinue,
}

impl<'r, 't> Nfa<'r, 't> {
    fn run(&mut self) -> CaptureLocs {
        let ncaps = match self.which {
            Exists => 0,
            Location => 1,
            Submatches => self.prog.num_captures(),
        };
        let mut matched = false;
        let ninsts = self.prog.insts.len();
        let mut clist = &mut Threads::new(self.which, ninsts, ncaps);
        let mut nlist = &mut Threads::new(self.which, ninsts, ncaps);

        let mut groups = Vec::from_elem(ncaps * 2, None);

        // Determine if the expression starts with a '^' so we can avoid
        // simulating .*?
        // Make sure multi-line mode isn't enabled for it, otherwise we can't
        // drop the initial .*?
        let prefix_anchor =
            match self.prog.insts[1] {
                EmptyBegin(flags) if flags & FLAG_MULTI == 0 => true,
                _ => false,
            };

        self.ic = self.start;
        let mut next_ic = self.chars.set(self.start);
        while self.ic <= self.end {
            if clist.size == 0 {
                // We have a match and we're done exploring alternatives.
                // Time to quit.
                if matched {
                    break
                }

                // If there are no threads to try, then we'll have to start
                // over at the beginning of the regex.
                // BUT, if there's a literal prefix for the program, try to
                // jump ahead quickly. If it can't be found, then we can bail
                // out early.
                if self.prog.prefix.len() > 0 && clist.size == 0 {
                    let needle = self.prog.prefix.as_slice().as_bytes();
                    let haystack = self.input.as_bytes().slice_from(self.ic);
                    match find_prefix(needle, haystack) {
                        None => break,
                        Some(i) => {
                            self.ic += i;
                            next_ic = self.chars.set(self.ic);
                        }
                    }
                }
            }

            // This simulates a preceding '.*?' for every regex by adding
            // a state starting at the current position in the input for the
            // beginning of the program only if we don't already have a match.
            if clist.size == 0 || (!prefix_anchor && !matched) {
                self.add(clist, 0, groups.as_mut_slice())
            }

            // Now we try to read the next character.
            // As a result, the 'step' method will look at the previous
            // character.
            self.ic = next_ic;
            next_ic = self.chars.advance();

            for i in range(0, clist.size) {
                let pc = clist.pc(i);
                let step_state = self.step(groups.as_mut_slice(), nlist,
                                           clist.groups(i), pc);
                match step_state {
                    StepMatchEarlyReturn => return vec![Some(0), Some(0)],
                    StepMatch => { matched = true; break },
                    StepContinue => {},
                }
            }
            mem::swap(&mut clist, &mut nlist);
            nlist.empty();
        }
        match self.which {
            Exists if matched     => vec![Some(0), Some(0)],
            Exists                => vec![None, None],
            Location | Submatches => groups,
        }
    }

    fn step(&self, groups: &mut [Option<uint>], nlist: &mut Threads,
            caps: &mut [Option<uint>], pc: uint)
           -> StepState {
        match self.prog.insts[pc] {
            Match => {
                match self.which {
                    Exists => {
                        return StepMatchEarlyReturn
                    }
                    Location => {
                        groups[0] = caps[0];
                        groups[1] = caps[1];
                        return StepMatch
                    }
                    Submatches => {
                        for (slot, val) in groups.mut_iter().zip(caps.iter()) {
                            *slot = *val;
                        }
                        return StepMatch
                    }
                }
            }
            OneChar(c, flags) => {
                if self.char_eq(flags & FLAG_NOCASE > 0, self.chars.prev, c) {
                    self.add(nlist, pc+1, caps);
                }
            }
            CharClass(ref ranges, flags) => {
                if self.chars.prev.is_some() {
                    let c = self.chars.prev.unwrap();
                    let negate = flags & FLAG_NEGATED > 0;
                    let casei = flags & FLAG_NOCASE > 0;
                    let found = ranges.as_slice();
                    let found = found.binary_search(|&rc| class_cmp(casei, c, rc))
                        .found().is_some();
                    if found ^ negate {
                        self.add(nlist, pc+1, caps);
                    }
                }
            }
            Any(flags) => {
                if flags & FLAG_DOTNL > 0
                   || !self.char_eq(false, self.chars.prev, '\n') {
                    self.add(nlist, pc+1, caps)
                }
            }
            EmptyBegin(_) | EmptyEnd(_) | EmptyWordBoundary(_)
            | Save(_) | Jump(_) | Split(_, _) => {},
        }
        StepContinue
    }

    fn add(&self, nlist: &mut Threads, pc: uint, groups: &mut [Option<uint>]) {
        if nlist.contains(pc) {
            return
        }
        // We have to add states to the threads list even if their empty.
        // TL;DR - It prevents cycles.
        // If we didn't care about cycles, we'd *only* add threads that
        // correspond to non-jumping instructions (OneChar, Any, Match, etc.).
        // But, it's possible for valid regexs (like '(a*)*') to result in
        // a cycle in the instruction list. e.g., We'll keep chasing the Split
        // instructions forever.
        // So we add these instructions to our thread queue, but in the main
        // VM loop, we look for them but simply ignore them.
        // Adding them to the queue prevents them from being revisited so we
        // can avoid cycles (and the inevitable stack overflow).
        //
        // We make a minor optimization by indicating that the state is "empty"
        // so that its capture groups are not filled in.
        match self.prog.insts[pc] {
            EmptyBegin(flags) => {
                let multi = flags & FLAG_MULTI > 0;
                nlist.add(pc, groups, true);
                if self.chars.is_begin()
                   || (multi && self.char_is(self.chars.prev, '\n')) {
                    self.add(nlist, pc + 1, groups)
                }
            }
            EmptyEnd(flags) => {
                let multi = flags & FLAG_MULTI > 0;
                nlist.add(pc, groups, true);
                if self.chars.is_end()
                   || (multi && self.char_is(self.chars.cur, '\n')) {
                    self.add(nlist, pc + 1, groups)
                }
            }
            EmptyWordBoundary(flags) => {
                nlist.add(pc, groups, true);
                if self.chars.is_word_boundary() == !(flags & FLAG_NEGATED > 0) {
                    self.add(nlist, pc + 1, groups)
                }
            }
            Save(slot) => {
                nlist.add(pc, groups, true);
                match self.which {
                    Location if slot <= 1 => {
                        let old = groups[slot];
                        groups[slot] = Some(self.ic);
                        self.add(nlist, pc + 1, groups);
                        groups[slot] = old;
                    }
                    Submatches => {
                        let old = groups[slot];
                        groups[slot] = Some(self.ic);
                        self.add(nlist, pc + 1, groups);
                        groups[slot] = old;
                    }
                    Exists | Location => self.add(nlist, pc + 1, groups),
                }
            }
            Jump(to) => {
                nlist.add(pc, groups, true);
                self.add(nlist, to, groups)
            }
            Split(x, y) => {
                nlist.add(pc, groups, true);
                self.add(nlist, x, groups);
                self.add(nlist, y, groups);
            }
            Match | OneChar(_, _) | CharClass(_, _) | Any(_) => {
                nlist.add(pc, groups, false);
            }
        }
    }

    // FIXME: For case insensitive comparisons, it uses the uppercase
    // character and tests for equality. IIUC, this does not generalize to
    // all of Unicode. I believe we need to check the entire fold for each
    // character. This will be easy to add if and when it gets added to Rust's
    // standard library.
    #[inline]
    fn char_eq(&self, casei: bool, textc: Option<char>, regc: char) -> bool {
        match textc {
            None => false,
            Some(textc) => {
                regc == textc
                    || (casei && regc.to_uppercase() == textc.to_uppercase())
            }
        }
    }

    #[inline]
    fn char_is(&self, textc: Option<char>, regc: char) -> bool {
        textc == Some(regc)
    }
}

/// CharReader is responsible for maintaining a "previous" and a "current"
/// character. This one-character lookahead is necessary for assertions that
/// look one character before or after the current position.
pub struct CharReader<'t> {
    /// The previous character read. It is None only when processing the first
    /// character of the input.
    pub prev: Option<char>,
    /// The current character.
    pub cur: Option<char>,
    input: &'t str,
    next: uint,
}

impl<'t> CharReader<'t> {
    /// Returns a new CharReader that advances through the input given.
    /// Note that a CharReader has no knowledge of the range in which to search
    /// the input.
    pub fn new(input: &'t str) -> CharReader<'t> {
        CharReader {
            prev: None,
            cur: None,
            input: input,
            next: 0,
       }
    }

    /// Sets the previous and current character given any arbitrary byte
    /// index (at a unicode codepoint boundary).
    #[inline]
    pub fn set(&mut self, ic: uint) -> uint {
        self.prev = None;
        self.cur = None;
        self.next = 0;

        if self.input.len() == 0 {
            return 1
        }
        if ic > 0 {
            let i = cmp::min(ic, self.input.len());
            let prev = self.input.char_range_at_reverse(i);
            self.prev = Some(prev.ch);
        }
        if ic < self.input.len() {
            let cur = self.input.char_range_at(ic);
            self.cur = Some(cur.ch);
            self.next = cur.next;
            self.next
        } else {
            self.input.len() + 1
        }
    }

    /// Does the same as `set`, except it always advances to the next
    /// character in the input (and therefore does half as many UTF8 decodings).
    #[inline]
    pub fn advance(&mut self) -> uint {
        self.prev = self.cur;
        if self.next < self.input.len() {
            let cur = self.input.char_range_at(self.next);
            self.cur = Some(cur.ch);
            self.next = cur.next;
        } else {
            self.cur = None;
            self.next = self.input.len() + 1;
        }
        self.next
    }

    /// Returns true if and only if this is the beginning of the input
    /// (ignoring the range of the input to search).
    #[inline]
    pub fn is_begin(&self) -> bool { self.prev.is_none() }

    /// Returns true if and only if this is the end of the input
    /// (ignoring the range of the input to search).
    #[inline]
    pub fn is_end(&self) -> bool { self.cur.is_none() }

    /// Returns true if and only if the current position is a word boundary.
    /// (Ignoring the range of the input to search.)
    pub fn is_word_boundary(&self) -> bool {
        if self.is_begin() {
            return is_word(self.cur)
        }
        if self.is_end() {
            return is_word(self.prev)
        }
        (is_word(self.cur) && !is_word(self.prev))
        || (is_word(self.prev) && !is_word(self.cur))
    }
}

struct Thread {
    pc: uint,
    groups: Vec<Option<uint>>,
}

struct Threads {
    which: MatchKind,
    queue: Vec<Thread>,
    sparse: Vec<uint>,
    size: uint,
}

impl Threads {
    // This is using a wicked neat trick to provide constant time lookup
    // for threads in the queue using a sparse set. A queue of threads is
    // allocated once with maximal size when the VM initializes and is reused
    // throughout execution. That is, there should be zero allocation during
    // the execution of a VM.
    //
    // See http://research.swtch.com/sparse for the deets.
    fn new(which: MatchKind, num_insts: uint, ncaps: uint) -> Threads {
        Threads {
            which: which,
            queue: Vec::from_fn(num_insts, |_| {
                Thread { pc: 0, groups: Vec::from_elem(ncaps * 2, None) }
            }),
            sparse: Vec::from_elem(num_insts, 0u),
            size: 0,
        }
    }

    fn add(&mut self, pc: uint, groups: &[Option<uint>], empty: bool) {
        let t = self.queue.get_mut(self.size);
        t.pc = pc;
        match (empty, self.which) {
            (_, Exists) | (true, _) => {},
            (false, Location) => {
                *t.groups.get_mut(0) = groups[0];
                *t.groups.get_mut(1) = groups[1];
            }
            (false, Submatches) => {
                for (slot, val) in t.groups.mut_iter().zip(groups.iter()) {
                    *slot = *val;
                }
            }
        }
        *self.sparse.get_mut(pc) = self.size;
        self.size += 1;
    }

    #[inline]
    fn contains(&self, pc: uint) -> bool {
        let s = self.sparse[pc];
        s < self.size && self.queue[s].pc == pc
    }

    #[inline]
    fn empty(&mut self) {
        self.size = 0;
    }

    #[inline]
    fn pc(&self, i: uint) -> uint {
        self.queue[i].pc
    }

    #[inline]
    fn groups<'r>(&'r mut self, i: uint) -> &'r mut [Option<uint>] {
        self.queue.get_mut(i).groups.as_mut_slice()
    }
}

/// Returns true if the character is a word character, according to the
/// (Unicode friendly) Perl character class '\w'.
/// Note that this is only use for testing word boundaries. The actual '\w'
/// is encoded as a CharClass instruction.
pub fn is_word(c: Option<char>) -> bool {
    let c = match c {
        None => return false,
        Some(c) => c,
    };
    // Try the common ASCII case before invoking binary search.
    match c {
        '_' | '0' .. '9' | 'a' .. 'z' | 'A' .. 'Z' => true,
        _ => PERLW.binary_search(|&(start, end)| {
            if c >= start && c <= end {
                Equal
            } else if start > c {
                Greater
            } else {
                Less
            }
        }).found().is_some()
    }
}

/// Given a character and a single character class range, return an ordering
/// indicating whether the character is less than the start of the range,
/// in the range (inclusive) or greater than the end of the range.
///
/// If `casei` is `true`, then this ordering is computed case insensitively.
///
/// This function is meant to be used with a binary search.
#[inline]
fn class_cmp(casei: bool, mut textc: char,
             (mut start, mut end): (char, char)) -> Ordering {
    if casei {
        // FIXME: This is pretty ridiculous. All of this case conversion
        // can be moved outside this function:
        // 1) textc should be uppercased outside the bsearch.
        // 2) the character class itself should be uppercased either in the
        //    parser or the compiler.
        // FIXME: This is too simplistic for correct Unicode support.
        //        See also: char_eq
        textc = textc.to_uppercase();
        start = start.to_uppercase();
        end = end.to_uppercase();
    }
    if textc >= start && textc <= end {
        Equal
    } else if start > textc {
        Greater
    } else {
        Less
    }
}

/// Returns the starting location of `needle` in `haystack`.
/// If `needle` is not in `haystack`, then `None` is returned.
///
/// Note that this is using a naive substring algorithm.
#[inline]
pub fn find_prefix(needle: &[u8], haystack: &[u8]) -> Option<uint> {
    let (hlen, nlen) = (haystack.len(), needle.len());
    if nlen > hlen || nlen == 0 {
        return None
    }
    for (offset, window) in haystack.windows(nlen).enumerate() {
        if window == needle {
            return Some(offset)
        }
    }
    None
}
