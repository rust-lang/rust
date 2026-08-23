//! This file constructs an implementation of the `Components` iterator that does not
//! consider prefix components and provides different implementation to specific functions
//! that utilize this version of `Components`.

use crate::cmp;
use crate::ffi::OsStr;
use crate::iter::FusedIterator;
use crate::path::{Component, MAIN_SEPARATOR, MAIN_SEPARATOR_STR, Path, PathBuf};
use crate::sys::path::is_sep_byte;

/// This tracks the current state of the iterator, or more specifically,
/// the bytes we have left to consume from our path.
///
/// We use:
///    - 'Absolute' to denote that our current path_bytes is considered an
///       an absolute path
///    - `Relative` to denote that our current path_bytes is considered a
///       relative path
///    - `Done` to denote that our current path_bytes is empty
///
/// All of these states allow us to parse the path components
/// appropriately within `Components::next`/`Components::next_back`.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
enum State {
    Absolute = 1, // A root component (i.e. '/')
    Relative = 2, // A relative component ('foo')
    Done = 3,     // Iterator is fully consumed
}

#[derive(Clone)]
pub struct Components<'a> {
    // The path left to parse components from
    path: &'a [u8],
    // The current state of the iterator
    state: State,
}

impl<'a> Components<'a> {
    /// Checks if all bytes of our path have been consumed
    #[inline]
    fn is_done(&self) -> bool {
        self.state == State::Done
    }

    /// This is the canonical implementation of `Path::has_root`
    #[inline]
    pub(crate) fn has_root(&self) -> bool {
        self.state == State::Absolute
    }

    /// Normalizes away trailing separators and current directory ('.') components
    /// in the forward direction. Returns the 0-index `self.path` should start at
    /// to subslice at in the front direction.
    fn normalize_front(&mut self, mut front: usize) -> usize {
        let mut cur_dir_present = false;
        match self.path[front..].iter().position(|b| {
            if !is_sep_byte(*b) {
                if *b == b'.' && !cur_dir_present {
                    cur_dir_present = true;
                    false
                } else {
                    true
                }
            } else {
                cur_dir_present = false;
                false
            }
        }) {
            None => {
                self.state = State::Done;
                return self.path.len();
            }
            Some(i) => {
                if cur_dir_present {
                    front += i - 1;
                } else {
                    front += i;
                }
            }
        }
        front
    }

    /// Normalizes away trailing separators and current directory ('.') components
    /// in the backward direction. Returns the 1-index `self.path` should start at
    /// to find next separator in the back direction.
    fn normalize_back(&mut self) -> usize {
        let mut cur_dir_present = false;
        match self.path.iter().rposition(|b| {
            if !is_sep_byte(*b) {
                if *b == b'.' && !cur_dir_present {
                    cur_dir_present = true;
                    false
                } else {
                    true
                }
            } else {
                cur_dir_present = false;
                false
            }
        }) {
            None => {
                // For cases like "./a", where our path
                // will observe "." at the end, and we need to return
                // that we observed "." component instead of
                // returning an empty path.
                if cur_dir_present {
                    1
                } else {
                    self.state = State::Done;
                    0
                }
            }
            Some(i) => {
                if cur_dir_present {
                    i + 2
                } else {
                    i + 1
                }
            }
        }
    }

    /// Extracts a slice corresponding to the portion of the path remaining for iteration.
    pub fn as_path(&self) -> &'a Path {
        // Normalize bytes from the right
        let mut cur_dir_present = false;
        let (done, back) = match self.path.iter().rposition(|b| {
            if !is_sep_byte(*b) {
                if *b == b'.' && !cur_dir_present {
                    cur_dir_present = true;
                    false
                } else {
                    true
                }
            } else {
                cur_dir_present = false;
                false
            }
        }) {
            None => {
                // For cases like "./a", where our path
                // will observe "." at the end, and we need to return
                // that we observed "." component instead of
                // returning an empty path.
                if cur_dir_present {
                    (false, 1)
                } else {
                    // self.is_done = true;
                    (true, 0)
                }
            }
            Some(i) => {
                if cur_dir_present {
                    (false, i + 2)
                } else {
                    (false, i + 1)
                }
            }
        };

        if done && self.has_root() {
            return Path::new("/");
        }
        // SAFETY: Back should be at a separator byte (or index 0 if
        // no separator byte exist), which slicing path_bytes at that index
        // should give us a valid slice
        unsafe { Path::from_u8_slice(&self.path[..back]) }
    }

    /// Parse a u8 slice into an OsStr, which is encoded into a `Component`
    fn parse_single_component(&self, slice: &'a [u8]) -> Option<Component<'a>> {
        match slice {
            [] => None,
            [b'.'] => Some(Component::CurDir),
            [b'.', b'.'] => Some(Component::ParentDir),
            _ => {
                let root_slice = MAIN_SEPARATOR_STR.as_bytes();
                if slice == root_slice {
                    return Some(Component::RootDir);
                }
                // SAFETY: Our sliced path is guaranteed to capture the entire component
                // due to delimiting on ascii separators from front and back.
                let path_osstr = unsafe { OsStr::from_encoded_bytes_unchecked(slice) };
                Some(Component::Normal(path_osstr))
            }
        }
    }

    /// Parses the next component in `Components<'_>` from the left. This returns both the index
    /// of the separator byte it saw (if None, it's the whole remaining slice) and the parsed
    /// component.
    fn parse_next_component(&mut self) -> (usize, Option<Component<'a>>) {
        let (front_ind, comp) = match self.path.iter().position(|b| is_sep_byte(*b)) {
            None => (self.path.len(), self.path),
            Some(i) => (i + 1, &self.path[..i]),
        };

        (front_ind, self.parse_single_component(comp))
    }

    /// Parses the next back component in `Components<'_>` from the right. This returns both the index
    /// of the separator byte it saw (if None, it's 0) and the parsed component.
    fn parse_next_back_component(&mut self, back: usize) -> (usize, Option<Component<'a>>) {
        let (back_ind, comp) = match self.path[..back].iter().rposition(|b| is_sep_byte(*b)) {
            None => {
                self.state = State::Done;
                (0, &self.path[..back])
            }
            Some(i) => (i, &self.path[i + 1..back]),
        };

        (back_ind, self.parse_single_component(comp))
    }
}

impl<'a> Iterator for Components<'a> {
    type Item = Component<'a>;

    fn next(&mut self) -> Option<Component<'a>> {
        // Changing this to a pure match case body with State::Absolute,
        // State::Relative, State::Done causes performance degradation
        // with `Components` ordering. Unsure why, but writing the code like
        // this maintains performance on par with the prefixed version.
        if !self.is_done() {
            match self.state {
                State::Absolute => {
                    let end_ind = self.normalize_front(0);
                    self.path = if self.is_done() {
                        &[]
                    } else {
                        self.state = State::Relative;
                        &self.path[end_ind..]
                    };

                    if !self.is_done() {
                        self.state = State::Relative;
                    }

                    return Some(Component::RootDir);
                }
                _ => {
                    let (front_ind, comp) = self.parse_next_component();
                    let normalized_front_ind = self.normalize_front(front_ind);
                    self.path =
                        if self.is_done() { &[] } else { &self.path[normalized_front_ind..] };
                    return comp;
                }
            }
        }
        None
    }
}

impl<'a> DoubleEndedIterator for Components<'a> {
    fn next_back(&mut self) -> Option<Component<'a>> {
        if !self.is_done() {
            let back = self.normalize_back();
            if self.is_done() {
                self.path = &[];
                return Some(Component::RootDir);
            } else {
                let (back_ind, comp) = self.parse_next_back_component(back);
                self.path = &self.path[..back_ind];
                return comp;
            }
        }

        None
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for Components<'_> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> PartialEq for Components<'a> {
    #[inline]
    fn eq(&self, other: &Components<'a>) -> bool {
        // Fast path for exact matches, e.g. for hashmap lookups.
        if self.path == other.path {
            return true;
        }

        eq_components(self.clone(), other.clone())
    }
}

/// Prefix-less `Components` equality
pub fn eq_components(mut left: Components<'_>, mut right: Components<'_>) -> bool {
    // Fast path
    //
    // - check if we have empty paths
    // - compare raw bytes from right to left to find first mismatch
    // - if first mismatch is not a separator or a curr dir component, return false directly. otherwise,
    //   backtrack to find separator after mismatch to avoid ambiguous parsings of '.' or '..' characters
    // - if found update state to only do a component-wise comparison in the back direction
    //   on the remainder, otherwise do it on the full path

    // One of them is an empty path
    if left.is_done() != right.is_done() {
        return false;
    }

    // Both are empty paths
    if left.is_done() && right.is_done() {
        return true;
    }

    let (left_diff, right_diff) = match left
        .path
        .iter()
        .rev()
        .zip(right.path.iter().rev())
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        None => {
            let bytes_consumed = left.path.len().min(right.path.len());
            (left.path.len() - bytes_consumed, right.path.len() - bytes_consumed)
        }
        Some((index, (a, b))) => {
            let a = *a;
            let b = *b;
            if a != b {
                if a != MAIN_SEPARATOR as u8 && a != b'.' && b != MAIN_SEPARATOR as u8 && b != b'.'
                {
                    return false;
                }
            }

            (left.path.len() - index - 1, right.path.len() - index - 1)
        }
    };

    // Cases like "foo/./bar" == "foo/bar", "foobar/bar" == "foobar", needs to consider
    // whether left_diff/right_diff is at a separator byte or not.
    if left.path[left_diff] != MAIN_SEPARATOR as u8 {
        if let Some(next_sep) = left.path[left_diff..].iter().position(|&b| is_sep_byte(b)) {
            left.path = &left.path[..left_diff + next_sep];
            right.path = &right.path[..right_diff + next_sep];
        }
    } else if right.path[right_diff] != MAIN_SEPARATOR as u8 {
        if let Some(next_sep) = right.path[right_diff..].iter().position(|&b| is_sep_byte(b)) {
            left.path = &left.path[..left_diff + next_sep];
            right.path = &right.path[..right_diff + next_sep];
        }
    } else {
        // We exclude the separator byte; if we happen to exclude the root dir component
        // for one of the paths (e.g. "///foo" == "/foo") that's fine, our
        // Component iterator state will still tell us that the path is an absolute
        // path, and then both would return `Component::RootDir`
        left.path = &left.path[..left_diff];
        right.path = &right.path[..right_diff];
    }

    // compare back to front since absolute paths often share long prefixes
    Iterator::eq(left.rev(), right.rev())
}

impl Eq for Components<'_> {}

impl<'a> PartialOrd for Components<'a> {
    #[inline]
    fn partial_cmp(&self, other: &Components<'a>) -> Option<cmp::Ordering> {
        Some(compare_components(self.clone(), other.clone()))
    }
}

impl Ord for Components<'_> {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        compare_components(self.clone(), other.clone())
    }
}

/// Prefix-less `Components` comparison
pub fn compare_components(mut left: Components<'_>, mut right: Components<'_>) -> cmp::Ordering {
    // Fast path for long shared prefixes
    //
    // - check for direct equality between both paths
    // - compare raw bytes to find first mismatch
    // - if first mismatch is not a separator or curr dir component, compare byte directly. otherwise,
    //   backtrack to find separator before mismatch to avoid ambiguous parsings of '.' or '..' characters
    // - if found update state to only do a component-wise comparison on the remainder,
    //   otherwise do it on the full path

    // Directly checking for equality should be fast, so this shouldn't incur
    // a major cost
    if left.path == right.path {
        return cmp::Ordering::Equal;
    }

    let first_difference =
        match left.path.iter().zip(right.path).enumerate().find(|(_, (a, b))| a != b) {
            None => left.path.len().min(right.path.len()),
            Some((index, (a, b))) => {
                let a = *a;
                let b = *b;

                if a != MAIN_SEPARATOR as u8 && a != b'.' && b != MAIN_SEPARATOR as u8 && b != b'.'
                {
                    return a.cmp(&b);
                }

                index
            }
        };

    if let Some(previous_sep) = left.path[..first_difference].iter().rposition(|&b| is_sep_byte(b))
    {
        // If our state initially started as an absolute path, the root component
        // is guaranteed to be sliced away, so treat the state as if it were a
        // relative component
        let mismatched_component_start = previous_sep + 1;
        left.state = State::Relative;
        right.state = State::Relative;
        left.path = &left.path[left.normalize_front(mismatched_component_start)..];
        right.path = &right.path[right.normalize_front(mismatched_component_start)..];
    }

    Iterator::cmp(left, right)
}

/// Prefix-less `Components` construction implementation
pub fn components(path: &Path) -> Components<'_> {
    let os_str_path = path.as_os_str();
    let path = os_str_path.as_encoded_bytes();

    let state = if path.is_empty() {
        State::Done
    } else if is_sep_byte(path[0]) {
        State::Absolute
    } else {
        State::Relative
    };

    Components { path, state }
}

/// Internal prefix-less helper method for `PathBuf::push`
pub fn push(self_path: &mut PathBuf, other_path: &Path) {
    // in general, a separator is needed if the rightmost byte is not a separator
    let buf = self_path.inner.as_encoded_bytes();
    let need_sep = buf.last().map(|c| !is_sep_byte(*c)).unwrap_or(false);
    let need_clear = other_path.is_absolute();

    // absolute `path` replaces `self`
    if need_clear {
        self_path.inner.clear();
    } else if other_path.has_root() {
        self_path.inner.truncate(0);
    // `path` is a pure relative path
    } else if need_sep {
        self_path.inner.push(MAIN_SEPARATOR_STR);
    }

    self_path.inner.push(other_path);
}
