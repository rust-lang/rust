use std::fmt;
use std::iter::FusedIterator;

/// Slot is a single saved capture location. Note that there are two slots for
/// every capture in a regular expression (one slot each for the start and end
/// of the capture).
pub type Slot = Option<usize>;

/// Locations represents the offsets of each capturing group in a regex for
/// a single match.
///
/// Unlike `Captures`, a `Locations` value only stores offsets.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct Locations(Vec<Slot>);

impl Locations {
    /// Returns the start and end positions of the Nth capture group. Returns
    /// `None` if `i` is not a valid capture group or if the capture group did
    /// not match anything. The positions returned are *always* byte indices
    /// with respect to the original string matched.
    pub fn pos(&self, i: usize) -> Option<(usize, usize)> {
        let (s, e) = (i.checked_mul(2)?, i.checked_mul(2)?.checked_add(1)?);
        match (self.0.get(s), self.0.get(e)) {
            (Some(&Some(s)), Some(&Some(e))) => Some((s, e)),
            _ => None,
        }
    }

    /// Creates an iterator of all the capture group positions in order of
    /// appearance in the regular expression. Positions are byte indices
    /// in terms of the original string matched.
    pub fn iter(&self) -> SubCapturesPosIter<'_> {
        SubCapturesPosIter { idx: 0, locs: self }
    }

    /// Returns the total number of capturing groups.
    ///
    /// This is always at least `1` since every regex has at least `1`
    /// capturing group that corresponds to the entire match.
    pub fn len(&self) -> usize {
        self.0.len() / 2
    }

    /// Return the individual slots as a slice.
    pub(crate) fn as_slots(&mut self) -> &mut [Slot] {
        &mut self.0
    }
}

/// An iterator over capture group positions for a particular match of a
/// regular expression.
///
/// Positions are byte indices in terms of the original string matched.
///
/// `'c` is the lifetime of the captures.
#[derive(Clone, Debug)]
pub struct SubCapturesPosIter<'c> {
    idx: usize,
    locs: &'c Locations,
}

impl<'c> Iterator for SubCapturesPosIter<'c> {
    type Item = Option<(usize, usize)>;

    fn next(&mut self) -> Option<Option<(usize, usize)>> {
        if self.idx >= self.locs.len() {
            return None;
        }
        let x = match self.locs.pos(self.idx) {
            None => Some(None),
            Some((s, e)) => Some(Some((s, e))),
        };
        self.idx += 1;
        x
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.locs.len() - self.idx;
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.len()
    }
}

impl<'c> ExactSizeIterator for SubCapturesPosIter<'c> {}

impl<'c> FusedIterator for SubCapturesPosIter<'c> {}

/// `RegularExpression` describes types that can implement regex searching.
///
/// This trait is my attempt at reducing code duplication and to standardize
/// the internal API. Specific duplication that is avoided are the `find`
/// and `capture` iterators, which are slightly tricky.
///
/// It's not clear whether this trait is worth it, and it also isn't
/// clear whether it's useful as a public trait or not. Methods like
/// `next_after_empty` reak of bad design, but the rest of the methods seem
/// somewhat reasonable. One particular thing this trait would expose would be
/// the ability to start the search of a regex anywhere in a haystack, which
/// isn't possible in the current public API.
pub trait RegularExpression: Sized + fmt::Debug {
    /// The type of the haystack.
    type Text: ?Sized + fmt::Debug;

    /// The number of capture slots in the compiled regular expression. This is
    /// always two times the number of capture groups (two slots per group).
    fn slots_len(&self) -> usize;

    /// Allocates fresh space for all capturing groups in this regex.
    fn locations(&self) -> Locations {
        Locations(vec![None; self.slots_len()])
    }

    /// Returns the position of the next character after `i`.
    ///
    /// For example, a haystack with type `&[u8]` probably returns `i+1`,
    /// whereas a haystack with type `&str` probably returns `i` plus the
    /// length of the next UTF-8 sequence.
    fn next_after_empty(&self, text: &Self::Text, i: usize) -> usize;

    /// Returns the location of the shortest match.
    fn shortest_match_at(
        &self,
        text: &Self::Text,
        start: usize,
    ) -> Option<usize>;

    /// Returns whether the regex matches the text given.
    fn is_match_at(&self, text: &Self::Text, start: usize) -> bool;

    /// Returns the leftmost-first match location if one exists.
    fn find_at(
        &self,
        text: &Self::Text,
        start: usize,
    ) -> Option<(usize, usize)>;

    /// Returns the leftmost-first match location if one exists, and also
    /// fills in any matching capture slot locations.
    fn captures_read_at(
        &self,
        locs: &mut Locations,
        text: &Self::Text,
        start: usize,
    ) -> Option<(usize, usize)>;

    /// Returns an iterator over all non-overlapping successive leftmost-first
    /// matches.
    fn find_iter(self, text: &Self::Text) -> Matches<'_, Self> {
        Matches { re: self, text, last_end: 0, last_match: None }
    }

    /// Returns an iterator over all non-overlapping successive leftmost-first
    /// matches with captures.
    fn captures_iter(self, text: &Self::Text) -> CaptureMatches<'_, Self> {
        CaptureMatches(self.find_iter(text))
    }
}

/// An iterator over all non-overlapping successive leftmost-first matches.
#[derive(Debug)]
pub struct Matches<'t, R>
where
    R: RegularExpression,
    R::Text: 't,
{
    re: R,
    text: &'t R::Text,
    last_end: usize,
    last_match: Option<usize>,
}

impl<'t, R> Matches<'t, R>
where
    R: RegularExpression,
    R::Text: 't,
{
    /// Return the text being searched.
    pub fn text(&self) -> &'t R::Text {
        self.text
    }

    /// Return the underlying regex.
    pub fn regex(&self) -> &R {
        &self.re
    }
}

impl<'t, R> Iterator for Matches<'t, R>
where
    R: RegularExpression,
    R::Text: 't + AsRef<[u8]>,
{
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.last_end > self.text.as_ref().len() {
            return None;
        }
        let (s, e) = match self.re.find_at(self.text, self.last_end) {
            None => return None,
            Some((s, e)) => (s, e),
        };
        if s == e {
            // This is an empty match. To ensure we make progress, start
            // the next search at the smallest possible starting position
            // of the next match following this one.
            self.last_end = self.re.next_after_empty(self.text, e);
            // Don't accept empty matches immediately following a match.
            // Just move on to the next match.
            if Some(e) == self.last_match {
                return self.next();
            }
        } else {
            self.last_end = e;
        }
        self.last_match = Some(e);
        Some((s, e))
    }
}

impl<'t, R> FusedIterator for Matches<'t, R>
where
    R: RegularExpression,
    R::Text: 't + AsRef<[u8]>,
{
}

/// An iterator over all non-overlapping successive leftmost-first matches with
/// captures.
#[derive(Debug)]
pub struct CaptureMatches<'t, R>(Matches<'t, R>)
where
    R: RegularExpression,
    R::Text: 't;

impl<'t, R> CaptureMatches<'t, R>
where
    R: RegularExpression,
    R::Text: 't,
{
    /// Return the text being searched.
    pub fn text(&self) -> &'t R::Text {
        self.0.text()
    }

    /// Return the underlying regex.
    pub fn regex(&self) -> &R {
        self.0.regex()
    }
}

impl<'t, R> Iterator for CaptureMatches<'t, R>
where
    R: RegularExpression,
    R::Text: 't + AsRef<[u8]>,
{
    type Item = Locations;

    fn next(&mut self) -> Option<Locations> {
        if self.0.last_end > self.0.text.as_ref().len() {
            return None;
        }
        let mut locs = self.0.re.locations();
        let (s, e) = match self.0.re.captures_read_at(
            &mut locs,
            self.0.text,
            self.0.last_end,
        ) {
            None => return None,
            Some((s, e)) => (s, e),
        };
        if s == e {
            self.0.last_end = self.0.re.next_after_empty(self.0.text, e);
            if Some(e) == self.0.last_match {
                return self.next();
            }
        } else {
            self.0.last_end = e;
        }
        self.0.last_match = Some(e);
        Some(locs)
    }
}

impl<'t, R> FusedIterator for CaptureMatches<'t, R>
where
    R: RegularExpression,
    R::Text: 't + AsRef<[u8]>,
{
}
