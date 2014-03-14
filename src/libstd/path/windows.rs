// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Windows file path handling

use ascii::AsciiCast;
use c_str::{CString, ToCStr};
use cast;
use clone::Clone;
use container::Container;
use cmp::Eq;
use from_str::FromStr;
use io::Writer;
use iter::{AdditiveIterator, DoubleEndedIterator, Extendable, Rev, Iterator, Map};
use option::{Option, Some, None};
use str;
use str::{CharSplits, OwnedStr, Str, StrVector, StrSlice};
use vec::{Vector, OwnedVector, ImmutableVector};
use super::{contains_nul, BytesContainer, GenericPath, GenericPathUnsafe};

/// Iterator that yields successive components of a Path as &str
///
/// Each component is yielded as Option<&str> for compatibility with PosixPath, but
/// every component in WindowsPath is guaranteed to be Some.
pub type StrComponents<'a> = Map<'a, &'a str, Option<&'a str>,
                                       CharSplits<'a, char>>;
/// Iterator that yields components of a Path in reverse as &str
///
/// Each component is yielded as Option<&str> for compatibility with PosixPath, but
/// every component in WindowsPath is guaranteed to be Some.
pub type RevStrComponents<'a> = Rev<Map<'a, &'a str, Option<&'a str>,
                                                 CharSplits<'a, char>>>;

/// Iterator that yields successive components of a Path as &[u8]
pub type Components<'a> = Map<'a, Option<&'a str>, &'a [u8],
                                    StrComponents<'a>>;
/// Iterator that yields components of a Path in reverse as &[u8]
pub type RevComponents<'a> = Map<'a, Option<&'a str>, &'a [u8],
                                       RevStrComponents<'a>>;

/// Represents a Windows path
// Notes for Windows path impl:
// The MAX_PATH is 260, but 253 is the practical limit due to some API bugs
// See http://msdn.microsoft.com/en-us/library/windows/desktop/aa365247.aspx for good information
// about windows paths.
// That same page puts a bunch of restrictions on allowed characters in a path.
// `\foo.txt` means "relative to current drive", but will not be considered to be absolute here
// as `∃P | P.join("\foo.txt") != "\foo.txt"`.
// `C:` is interesting, that means "the current directory on drive C".
// Long absolute paths need to have \\?\ prefix (or, for UNC, \\?\UNC\). I think that can be
// ignored for now, though, and only added in a hypothetical .to_pwstr() function.
// However, if a path is parsed that has \\?\, this needs to be preserved as it disables the
// processing of "." and ".." components and / as a separator.
// Experimentally, \\?\foo is not the same thing as \foo.
// Also, \\foo is not valid either (certainly not equivalent to \foo).
// Similarly, C:\\Users is not equivalent to C:\Users, although C:\Users\\foo is equivalent
// to C:\Users\foo. In fact the command prompt treats C:\\foo\bar as UNC path. But it might be
// best to just ignore that and normalize it to C:\foo\bar.
//
// Based on all this, I think the right approach is to do the following:
// * Require valid utf-8 paths. Windows API may use WCHARs, but we don't, and utf-8 is convertible
// to UTF-16 anyway (though does Windows use UTF-16 or UCS-2? Not sure).
// * Parse the prefixes \\?\UNC\, \\?\, and \\.\ explicitly.
// * If \\?\UNC\, treat following two path components as server\share. Don't error for missing
//   server\share.
// * If \\?\, parse disk from following component, if present. Don't error for missing disk.
// * If \\.\, treat rest of path as just regular components. I don't know how . and .. are handled
//   here, they probably aren't, but I'm not going to worry about that.
// * Else if starts with \\, treat following two components as server\share. Don't error for missing
//   server\share.
// * Otherwise, attempt to parse drive from start of path.
//
// The only error condition imposed here is valid utf-8. All other invalid paths are simply
// preserved by the data structure; let the Windows API error out on them.
#[deriving(Clone)]
pub struct Path {
    priv repr: ~str, // assumed to never be empty
    priv prefix: Option<PathPrefix>,
    priv sepidx: Option<uint> // index of the final separator in the non-prefix portion of repr
}

impl Eq for Path {
    #[inline]
    fn eq(&self, other: &Path) -> bool {
        self.repr == other.repr
    }
}

impl FromStr for Path {
    fn from_str(s: &str) -> Option<Path> {
        Path::new_opt(s)
    }
}

impl ToCStr for Path {
    #[inline]
    fn to_c_str(&self) -> CString {
        // The Path impl guarantees no embedded NULs
        unsafe { self.as_vec().to_c_str_unchecked() }
    }

    #[inline]
    unsafe fn to_c_str_unchecked(&self) -> CString {
        self.as_vec().to_c_str_unchecked()
    }
}

impl<S: Writer> ::hash::Hash<S> for Path {
    #[inline]
    fn hash(&self, state: &mut S) {
        self.repr.hash(state)
    }
}

impl BytesContainer for Path {
    #[inline]
    fn container_as_bytes<'a>(&'a self) -> &'a [u8] {
        self.as_vec()
    }
    #[inline]
    fn container_into_owned_bytes(self) -> ~[u8] {
        self.into_vec()
    }
    #[inline]
    fn container_as_str<'a>(&'a self) -> Option<&'a str> {
        self.as_str()
    }
    #[inline]
    fn is_str(_: Option<Path>) -> bool { true }
}

impl<'a> BytesContainer for &'a Path {
    #[inline]
    fn container_as_bytes<'a>(&'a self) -> &'a [u8] {
        self.as_vec()
    }
    #[inline]
    fn container_as_str<'a>(&'a self) -> Option<&'a str> {
        self.as_str()
    }
    #[inline]
    fn is_str(_: Option<&'a Path>) -> bool { true }
}

impl GenericPathUnsafe for Path {
    /// See `GenericPathUnsafe::from_vec_unchecked`.
    ///
    /// # Failure
    ///
    /// Fails if not valid UTF-8.
    #[inline]
    unsafe fn new_unchecked<T: BytesContainer>(path: T) -> Path {
        let (prefix, path) = Path::normalize_(path.container_as_str().unwrap());
        assert!(!path.is_empty());
        let mut ret = Path{ repr: path, prefix: prefix, sepidx: None };
        ret.update_sepidx();
        ret
    }

    /// See `GenericPathUnsafe::set_filename_unchecekd`.
    ///
    /// # Failure
    ///
    /// Fails if not valid UTF-8.
    unsafe fn set_filename_unchecked<T: BytesContainer>(&mut self, filename: T) {
        let filename = filename.container_as_str().unwrap();
        match self.sepidx_or_prefix_len() {
            None if ".." == self.repr => {
                let mut s = str::with_capacity(3 + filename.len());
                s.push_str("..");
                s.push_char(SEP);
                s.push_str(filename);
                self.update_normalized(s);
            }
            None => {
                self.update_normalized(filename);
            }
            Some((_,idxa,end)) if self.repr.slice(idxa,end) == ".." => {
                let mut s = str::with_capacity(end + 1 + filename.len());
                s.push_str(self.repr.slice_to(end));
                s.push_char(SEP);
                s.push_str(filename);
                self.update_normalized(s);
            }
            Some((idxb,idxa,_)) if self.prefix == Some(DiskPrefix) && idxa == self.prefix_len() => {
                let mut s = str::with_capacity(idxb + filename.len());
                s.push_str(self.repr.slice_to(idxb));
                s.push_str(filename);
                self.update_normalized(s);
            }
            Some((idxb,_,_)) => {
                let mut s = str::with_capacity(idxb + 1 + filename.len());
                s.push_str(self.repr.slice_to(idxb));
                s.push_char(SEP);
                s.push_str(filename);
                self.update_normalized(s);
            }
        }
    }

    /// See `GenericPathUnsafe::push_unchecked`.
    ///
    /// Concatenating two Windows Paths is rather complicated.
    /// For the most part, it will behave as expected, except in the case of
    /// pushing a volume-relative path, e.g. `C:foo.txt`. Because we have no
    /// concept of per-volume cwds like Windows does, we can't behave exactly
    /// like Windows will. Instead, if the receiver is an absolute path on
    /// the same volume as the new path, it will be treated as the cwd that
    /// the new path is relative to. Otherwise, the new path will be treated
    /// as if it were absolute and will replace the receiver outright.
    unsafe fn push_unchecked<T: BytesContainer>(&mut self, path: T) {
        let path = path.container_as_str().unwrap();
        fn is_vol_abs(path: &str, prefix: Option<PathPrefix>) -> bool {
            // assume prefix is Some(DiskPrefix)
            let rest = path.slice_from(prefix_len(prefix));
            !rest.is_empty() && rest[0].is_ascii() && is_sep(rest[0] as char)
        }
        fn shares_volume(me: &Path, path: &str) -> bool {
            // path is assumed to have a prefix of Some(DiskPrefix)
            match me.prefix {
                Some(DiskPrefix) => me.repr[0] == path[0].to_ascii().to_upper().to_byte(),
                Some(VerbatimDiskPrefix) => me.repr[4] == path[0].to_ascii().to_upper().to_byte(),
                _ => false
            }
        }
        fn is_sep_(prefix: Option<PathPrefix>, u: u8) -> bool {
            if prefix_is_verbatim(prefix) { is_sep_verbatim(u as char) }
            else { is_sep(u as char) }
        }

        fn replace_path(me: &mut Path, path: &str, prefix: Option<PathPrefix>) {
            let newpath = Path::normalize__(path, prefix);
            me.repr = match newpath {
                Some(p) => p,
                None => path.to_owned()
            };
            me.prefix = prefix;
            me.update_sepidx();
        }
        fn append_path(me: &mut Path, path: &str) {
            // appends a path that has no prefix
            // if me is verbatim, we need to pre-normalize the new path
            let path_ = if is_verbatim(me) { Path::normalize__(path, None) }
                        else { None };
            let pathlen = path_.as_ref().map_or(path.len(), |p| p.len());
            let mut s = str::with_capacity(me.repr.len() + 1 + pathlen);
            s.push_str(me.repr);
            let plen = me.prefix_len();
            // if me is "C:" we don't want to add a path separator
            match me.prefix {
                Some(DiskPrefix) if me.repr.len() == plen => (),
                _ if !(me.repr.len() > plen && me.repr[me.repr.len()-1] == SEP_BYTE) => {
                    s.push_char(SEP);
                }
                _ => ()
            }
            match path_ {
                None => s.push_str(path),
                Some(p) => s.push_str(p)
            };
            me.update_normalized(s)
        }

        if !path.is_empty() {
            let prefix = parse_prefix(path);
            match prefix {
                Some(DiskPrefix) if !is_vol_abs(path, prefix) && shares_volume(self, path) => {
                    // cwd-relative path, self is on the same volume
                    append_path(self, path.slice_from(prefix_len(prefix)));
                }
                Some(_) => {
                    // absolute path, or cwd-relative and self is not same volume
                    replace_path(self, path, prefix);
                }
                None if !path.is_empty() && is_sep_(self.prefix, path[0]) => {
                    // volume-relative path
                    if self.prefix.is_some() {
                        // truncate self down to the prefix, then append
                        let n = self.prefix_len();
                        self.repr.truncate(n);
                        append_path(self, path);
                    } else {
                        // we have no prefix, so nothing to be relative to
                        replace_path(self, path, prefix);
                    }
                }
                None => {
                    // relative path
                    append_path(self, path);
                }
            }
        }
    }
}

impl GenericPath for Path {
    #[inline]
    fn new_opt<T: BytesContainer>(path: T) -> Option<Path> {
        let s = path.container_as_str();
        match s {
            None => None,
            Some(s) => {
                if contains_nul(s.as_bytes()) {
                    None
                } else {
                    Some(unsafe { GenericPathUnsafe::new_unchecked(s) })
                }
            }
        }
    }

    /// See `GenericPath::as_str` for info.
    /// Always returns a `Some` value.
    #[inline]
    fn as_str<'a>(&'a self) -> Option<&'a str> {
        Some(self.repr.as_slice())
    }

    #[inline]
    fn as_vec<'a>(&'a self) -> &'a [u8] {
        self.repr.as_bytes()
    }

    #[inline]
    fn into_vec(self) -> ~[u8] {
        self.repr.into_bytes()
    }

    #[inline]
    fn dirname<'a>(&'a self) -> &'a [u8] {
        self.dirname_str().unwrap().as_bytes()
    }

    /// See `GenericPath::dirname_str` for info.
    /// Always returns a `Some` value.
    fn dirname_str<'a>(&'a self) -> Option<&'a str> {
        Some(match self.sepidx_or_prefix_len() {
            None if ".." == self.repr => self.repr.as_slice(),
            None => ".",
            Some((_,idxa,end)) if self.repr.slice(idxa, end) == ".." => {
                self.repr.as_slice()
            }
            Some((idxb,_,end)) if self.repr.slice(idxb, end) == "\\" => {
                self.repr.as_slice()
            }
            Some((0,idxa,_)) => self.repr.slice_to(idxa),
            Some((idxb,idxa,_)) => {
                match self.prefix {
                    Some(DiskPrefix) | Some(VerbatimDiskPrefix) if idxb == self.prefix_len() => {
                        self.repr.slice_to(idxa)
                    }
                    _ => self.repr.slice_to(idxb)
                }
            }
        })
    }

    #[inline]
    fn filename<'a>(&'a self) -> Option<&'a [u8]> {
        self.filename_str().map(|x| x.as_bytes())
    }

    /// See `GenericPath::filename_str` for info.
    /// Always returns a `Some` value if `filename` returns a `Some` value.
    fn filename_str<'a>(&'a self) -> Option<&'a str> {
        match self.sepidx_or_prefix_len() {
            None if "." == self.repr || ".." == self.repr => None,
            None => Some(self.repr.as_slice()),
            Some((_,idxa,end)) if self.repr.slice(idxa, end) == ".." => None,
            Some((_,idxa,end)) if idxa == end => None,
            Some((_,idxa,end)) => Some(self.repr.slice(idxa, end))
        }
    }

    /// See `GenericPath::filestem_str` for info.
    /// Always returns a `Some` value if `filestem` returns a `Some` value.
    #[inline]
    fn filestem_str<'a>(&'a self) -> Option<&'a str> {
        // filestem() returns a byte vector that's guaranteed valid UTF-8
        self.filestem().map(|t| unsafe { cast::transmute(t) })
    }

    #[inline]
    fn extension_str<'a>(&'a self) -> Option<&'a str> {
        // extension() returns a byte vector that's guaranteed valid UTF-8
        self.extension().map(|t| unsafe { cast::transmute(t) })
    }

    fn dir_path(&self) -> Path {
        unsafe { GenericPathUnsafe::new_unchecked(self.dirname_str().unwrap()) }
    }

    #[inline]
    fn pop(&mut self) -> bool {
        match self.sepidx_or_prefix_len() {
            None if "." == self.repr => false,
            None => {
                self.repr = ~".";
                self.sepidx = None;
                true
            }
            Some((idxb,idxa,end)) if idxb == idxa && idxb == end => false,
            Some((idxb,_,end)) if self.repr.slice(idxb, end) == "\\" => false,
            Some((idxb,idxa,_)) => {
                let trunc = match self.prefix {
                    Some(DiskPrefix) | Some(VerbatimDiskPrefix) | None => {
                        let plen = self.prefix_len();
                        if idxb == plen { idxa } else { idxb }
                    }
                    _ => idxb
                };
                self.repr.truncate(trunc);
                self.update_sepidx();
                true
            }
        }
    }

    fn root_path(&self) -> Option<Path> {
        if self.prefix.is_some() {
            Some(Path::new(match self.prefix {
                Some(DiskPrefix) if self.is_absolute() => {
                    self.repr.slice_to(self.prefix_len()+1)
                }
                Some(VerbatimDiskPrefix) => {
                    self.repr.slice_to(self.prefix_len()+1)
                }
                _ => self.repr.slice_to(self.prefix_len())
            }))
        } else if is_vol_relative(self) {
            Some(Path::new(self.repr.slice_to(1)))
        } else {
            None
        }
    }

    /// See `GenericPath::is_absolute` for info.
    ///
    /// A Windows Path is considered absolute only if it has a non-volume prefix,
    /// or if it has a volume prefix and the path starts with '\'.
    /// A path of `\foo` is not considered absolute because it's actually
    /// relative to the "current volume". A separate method `Path::is_vol_relative`
    /// is provided to indicate this case. Similarly a path of `C:foo` is not
    /// considered absolute because it's relative to the cwd on volume C:. A
    /// separate method `Path::is_cwd_relative` is provided to indicate this case.
    #[inline]
    fn is_absolute(&self) -> bool {
        match self.prefix {
            Some(DiskPrefix) => {
                let rest = self.repr.slice_from(self.prefix_len());
                rest.len() > 0 && rest[0] == SEP_BYTE
            }
            Some(_) => true,
            None => false
        }
    }

    #[inline]
    fn is_relative(&self) -> bool {
        self.prefix.is_none() && !is_vol_relative(self)
    }

    fn is_ancestor_of(&self, other: &Path) -> bool {
        if !self.equiv_prefix(other) {
            false
        } else if self.is_absolute() != other.is_absolute() ||
                  is_vol_relative(self) != is_vol_relative(other) {
            false
        } else {
            let mut ita = self.str_components().map(|x|x.unwrap());
            let mut itb = other.str_components().map(|x|x.unwrap());
            if "." == self.repr {
                return itb.next() != Some("..");
            }
            loop {
                match (ita.next(), itb.next()) {
                    (None, _) => break,
                    (Some(a), Some(b)) if a == b => { continue },
                    (Some(a), _) if a == ".." => {
                        // if ita contains only .. components, it's an ancestor
                        return ita.all(|x| x == "..");
                    }
                    _ => return false
                }
            }
            true
        }
    }

    fn path_relative_from(&self, base: &Path) -> Option<Path> {
        fn comp_requires_verbatim(s: &str) -> bool {
            s == "." || s == ".." || s.contains_char(SEP2)
        }

        if !self.equiv_prefix(base) {
            // prefixes differ
            if self.is_absolute() {
                Some(self.clone())
            } else if self.prefix == Some(DiskPrefix) && base.prefix == Some(DiskPrefix) {
                // both drives, drive letters must differ or they'd be equiv
                Some(self.clone())
            } else {
                None
            }
        } else if self.is_absolute() != base.is_absolute() {
            if self.is_absolute() {
                Some(self.clone())
            } else {
                None
            }
        } else if is_vol_relative(self) != is_vol_relative(base) {
            if is_vol_relative(self) {
                Some(self.clone())
            } else {
                None
            }
        } else {
            let mut ita = self.str_components().map(|x|x.unwrap());
            let mut itb = base.str_components().map(|x|x.unwrap());
            let mut comps = ~[];

            let a_verb = is_verbatim(self);
            let b_verb = is_verbatim(base);
            loop {
                match (ita.next(), itb.next()) {
                    (None, None) => break,
                    (Some(a), None) if a_verb && comp_requires_verbatim(a) => {
                        return Some(self.clone())
                    }
                    (Some(a), None) => {
                        comps.push(a);
                        if !a_verb {
                            comps.extend(&mut ita);
                            break;
                        }
                    }
                    (None, _) => comps.push(".."),
                    (Some(a), Some(b)) if comps.is_empty() && a == b => (),
                    (Some(a), Some(b)) if !b_verb && b == "." => {
                        if a_verb && comp_requires_verbatim(a) {
                            return Some(self.clone())
                        } else { comps.push(a) }
                    }
                    (Some(_), Some(b)) if !b_verb && b == ".." => return None,
                    (Some(a), Some(_)) if a_verb && comp_requires_verbatim(a) => {
                        return Some(self.clone())
                    }
                    (Some(a), Some(_)) => {
                        comps.push("..");
                        for _ in itb {
                            comps.push("..");
                        }
                        comps.push(a);
                        if !a_verb {
                            comps.extend(&mut ita);
                            break;
                        }
                    }
                }
            }
            Some(Path::new(comps.connect("\\")))
        }
    }

    fn ends_with_path(&self, child: &Path) -> bool {
        if !child.is_relative() { return false; }
        let mut selfit = self.str_components().rev();
        let mut childit = child.str_components().rev();
        loop {
            match (selfit.next(), childit.next()) {
                (Some(a), Some(b)) => if a != b { return false; },
                (Some(_), None) => break,
                (None, Some(_)) => return false,
                (None, None) => break
            }
        }
        true
    }
}

impl Path {
    /// Returns a new Path from a byte vector or string
    ///
    /// # Failure
    ///
    /// Fails the task if the vector contains a NUL.
    /// Fails if invalid UTF-8.
    #[inline]
    pub fn new<T: BytesContainer>(path: T) -> Path {
        GenericPath::new(path)
    }

    /// Returns a new Path from a byte vector or string, if possible
    #[inline]
    pub fn new_opt<T: BytesContainer>(path: T) -> Option<Path> {
        GenericPath::new_opt(path)
    }

    /// Returns an iterator that yields each component of the path in turn as a Option<&str>.
    /// Every component is guaranteed to be Some.
    /// Does not yield the path prefix (including server/share components in UNC paths).
    /// Does not distinguish between volume-relative and relative paths, e.g.
    /// \a\b\c and a\b\c.
    /// Does not distinguish between absolute and cwd-relative paths, e.g.
    /// C:\foo and C:foo.
    pub fn str_components<'a>(&'a self) -> StrComponents<'a> {
        let s = match self.prefix {
            Some(_) => {
                let plen = self.prefix_len();
                if self.repr.len() > plen && self.repr[plen] == SEP_BYTE {
                    self.repr.slice_from(plen+1)
                } else { self.repr.slice_from(plen) }
            }
            None if self.repr[0] == SEP_BYTE => self.repr.slice_from(1),
            None => self.repr.as_slice()
        };
        let ret = s.split_terminator(SEP).map(Some);
        ret
    }

    /// Returns an iterator that yields each component of the path in reverse as an Option<&str>
    /// See str_components() for details.
    pub fn rev_str_components<'a>(&'a self) -> RevStrComponents<'a> {
        self.str_components().rev()
    }

    /// Returns an iterator that yields each component of the path in turn as a &[u8].
    /// See str_components() for details.
    pub fn components<'a>(&'a self) -> Components<'a> {
        fn convert<'a>(x: Option<&'a str>) -> &'a [u8] {
            #[inline];
            x.unwrap().as_bytes()
        }
        self.str_components().map(convert)
    }

    /// Returns an iterator that yields each component of the path in reverse as a &[u8].
    /// See str_components() for details.
    pub fn rev_components<'a>(&'a self) -> RevComponents<'a> {
        fn convert<'a>(x: Option<&'a str>) -> &'a [u8] {
            #[inline];
            x.unwrap().as_bytes()
        }
        self.rev_str_components().map(convert)
    }

    fn equiv_prefix(&self, other: &Path) -> bool {
        match (self.prefix, other.prefix) {
            (Some(DiskPrefix), Some(VerbatimDiskPrefix)) => {
                self.is_absolute() &&
                    self.repr[0].to_ascii().eq_ignore_case(other.repr[4].to_ascii())
            }
            (Some(VerbatimDiskPrefix), Some(DiskPrefix)) => {
                other.is_absolute() &&
                    self.repr[4].to_ascii().eq_ignore_case(other.repr[0].to_ascii())
            }
            (Some(VerbatimDiskPrefix), Some(VerbatimDiskPrefix)) => {
                self.repr[4].to_ascii().eq_ignore_case(other.repr[4].to_ascii())
            }
            (Some(UNCPrefix(_,_)), Some(VerbatimUNCPrefix(_,_))) => {
                self.repr.slice(2, self.prefix_len()) == other.repr.slice(8, other.prefix_len())
            }
            (Some(VerbatimUNCPrefix(_,_)), Some(UNCPrefix(_,_))) => {
                self.repr.slice(8, self.prefix_len()) == other.repr.slice(2, other.prefix_len())
            }
            (None, None) => true,
            (a, b) if a == b => {
                self.repr.slice_to(self.prefix_len()) == other.repr.slice_to(other.prefix_len())
            }
            _ => false
        }
    }

    fn normalize_<S: Str>(s: S) -> (Option<PathPrefix>, ~str) {
        // make borrowck happy
        let (prefix, val) = {
            let prefix = parse_prefix(s.as_slice());
            let path = Path::normalize__(s.as_slice(), prefix);
            (prefix, path)
        };
        (prefix, match val {
            None => s.into_owned(),
            Some(val) => val
        })
    }

    fn normalize__(s: &str, prefix: Option<PathPrefix>) -> Option<~str> {
        if prefix_is_verbatim(prefix) {
            // don't do any normalization
            match prefix {
                Some(VerbatimUNCPrefix(x, 0)) if s.len() == 8 + x => {
                    // the server component has no trailing '\'
                    let mut s = s.into_owned();
                    s.push_char(SEP);
                    Some(s)
                }
                _ => None
            }
        } else {
            let (is_abs, comps) = normalize_helper(s, prefix);
            let mut comps = comps;
            match (comps.is_some(),prefix) {
                (false, Some(DiskPrefix)) => {
                    if s[0] >= 'a' as u8 && s[0] <= 'z' as u8 {
                        comps = Some(~[]);
                    }
                }
                (false, Some(VerbatimDiskPrefix)) => {
                    if s[4] >= 'a' as u8 && s[0] <= 'z' as u8 {
                        comps = Some(~[]);
                    }
                }
                _ => ()
            }
            match comps {
                None => None,
                Some(comps) => {
                    if prefix.is_some() && comps.is_empty() {
                        match prefix.unwrap() {
                            DiskPrefix => {
                                let len = prefix_len(prefix) + is_abs as uint;
                                let mut s = s.slice_to(len).to_owned();
                                unsafe {
                                    str::raw::as_owned_vec(&mut s)[0] =
                                        s[0].to_ascii().to_upper().to_byte();
                                }
                                if is_abs {
                                    // normalize C:/ to C:\
                                    unsafe {
                                        str::raw::as_owned_vec(&mut s)[2] = SEP_BYTE;
                                    }
                                }
                                Some(s)
                            }
                            VerbatimDiskPrefix => {
                                let len = prefix_len(prefix) + is_abs as uint;
                                let mut s = s.slice_to(len).to_owned();
                                unsafe {
                                    str::raw::as_owned_vec(&mut s)[4] =
                                        s[4].to_ascii().to_upper().to_byte();
                                }
                                Some(s)
                            }
                            _ => {
                                let plen = prefix_len(prefix);
                                if s.len() > plen {
                                    Some(s.slice_to(plen).to_owned())
                                } else { None }
                            }
                        }
                    } else if is_abs && comps.is_empty() {
                        Some(str::from_char(SEP))
                    } else {
                        let prefix_ = s.slice_to(prefix_len(prefix));
                        let n = prefix_.len() +
                                if is_abs { comps.len() } else { comps.len() - 1} +
                                comps.iter().map(|v| v.len()).sum();
                        let mut s = str::with_capacity(n);
                        match prefix {
                            Some(DiskPrefix) => {
                                s.push_char(prefix_[0].to_ascii().to_upper().to_char());
                                s.push_char(':');
                            }
                            Some(VerbatimDiskPrefix) => {
                                s.push_str(prefix_.slice_to(4));
                                s.push_char(prefix_[4].to_ascii().to_upper().to_char());
                                s.push_str(prefix_.slice_from(5));
                            }
                            Some(UNCPrefix(a,b)) => {
                                s.push_str("\\\\");
                                s.push_str(prefix_.slice(2, a+2));
                                s.push_char(SEP);
                                s.push_str(prefix_.slice(3+a, 3+a+b));
                            }
                            Some(_) => s.push_str(prefix_),
                            None => ()
                        }
                        let mut it = comps.move_iter();
                        if !is_abs {
                            match it.next() {
                                None => (),
                                Some(comp) => s.push_str(comp)
                            }
                        }
                        for comp in it {
                            s.push_char(SEP);
                            s.push_str(comp);
                        }
                        Some(s)
                    }
                }
            }
        }
    }

    fn update_sepidx(&mut self) {
        let s = if self.has_nonsemantic_trailing_slash() {
                    self.repr.slice_to(self.repr.len()-1)
                } else { self.repr.as_slice() };
        let idx = s.rfind(if !prefix_is_verbatim(self.prefix) { is_sep }
                          else { is_sep_verbatim });
        let prefixlen = self.prefix_len();
        self.sepidx = idx.and_then(|x| if x < prefixlen { None } else { Some(x) });
    }

    fn prefix_len(&self) -> uint {
        prefix_len(self.prefix)
    }

    // Returns a tuple (before, after, end) where before is the index of the separator
    // and after is the index just after the separator.
    // end is the length of the string, normally, or the index of the final character if it is
    // a non-semantic trailing separator in a verbatim string.
    // If the prefix is considered the separator, before and after are the same.
    fn sepidx_or_prefix_len(&self) -> Option<(uint,uint,uint)> {
        match self.sepidx {
            None => match self.prefix_len() { 0 => None, x => Some((x,x,self.repr.len())) },
            Some(x) => {
                if self.has_nonsemantic_trailing_slash() {
                    Some((x,x+1,self.repr.len()-1))
                } else { Some((x,x+1,self.repr.len())) }
            }
        }
    }

    fn has_nonsemantic_trailing_slash(&self) -> bool {
        is_verbatim(self) && self.repr.len() > self.prefix_len()+1 &&
            self.repr[self.repr.len()-1] == SEP_BYTE
    }

    fn update_normalized<S: Str>(&mut self, s: S) {
        let (prefix, path) = Path::normalize_(s);
        self.repr = path;
        self.prefix = prefix;
        self.update_sepidx();
    }
}

/// Returns whether the path is considered "volume-relative", which means a path
/// that looks like "\foo". Paths of this form are relative to the current volume,
/// but absolute within that volume.
#[inline]
pub fn is_vol_relative(path: &Path) -> bool {
    path.prefix.is_none() && is_sep_byte(&path.repr[0])
}

/// Returns whether the path is considered "cwd-relative", which means a path
/// with a volume prefix that is not absolute. This look like "C:foo.txt". Paths
/// of this form are relative to the cwd on the given volume.
#[inline]
pub fn is_cwd_relative(path: &Path) -> bool {
    path.prefix == Some(DiskPrefix) && !path.is_absolute()
}

/// Returns the PathPrefix for this Path
#[inline]
pub fn prefix(path: &Path) -> Option<PathPrefix> {
    path.prefix
}

/// Returns whether the Path's prefix is a verbatim prefix, i.e. `\\?\`
#[inline]
pub fn is_verbatim(path: &Path) -> bool {
    prefix_is_verbatim(path.prefix)
}

/// Returns the non-verbatim equivalent of the input path, if possible.
/// If the input path is a device namespace path, None is returned.
/// If the input path is not verbatim, it is returned as-is.
/// If the input path is verbatim, but the same path can be expressed as
/// non-verbatim, the non-verbatim version is returned.
/// Otherwise, None is returned.
pub fn make_non_verbatim(path: &Path) -> Option<Path> {
    let new_path = match path.prefix {
        Some(VerbatimPrefix(_)) | Some(DeviceNSPrefix(_)) => return None,
        Some(UNCPrefix(_,_)) | Some(DiskPrefix) | None => return Some(path.clone()),
        Some(VerbatimDiskPrefix) => {
            // \\?\D:\
            Path::new(path.repr.slice_from(4))
        }
        Some(VerbatimUNCPrefix(_,_)) => {
            // \\?\UNC\server\share
            Path::new(format!(r"\\{}", path.repr.slice_from(7)))
        }
    };
    if new_path.prefix.is_none() {
        // \\?\UNC\server is a VerbatimUNCPrefix
        // but \\server is nothing
        return None;
    }
    // now ensure normalization didn't change anything
    if path.repr.slice_from(path.prefix_len()) == new_path.repr.slice_from(new_path.prefix_len()) {
        Some(new_path)
    } else {
        None
    }
}

/// The standard path separator character
pub static SEP: char = '\\';
/// The standard path separator byte
pub static SEP_BYTE: u8 = SEP as u8;

/// The alternative path separator character
pub static SEP2: char = '/';
/// The alternative path separator character
pub static SEP2_BYTE: u8 = SEP2 as u8;

/// Returns whether the given char is a path separator.
/// Allows both the primary separator '\' and the alternative separator '/'.
#[inline]
pub fn is_sep(c: char) -> bool {
    c == SEP || c == SEP2
}

/// Returns whether the given char is a path separator.
/// Only allows the primary separator '\'; use is_sep to allow '/'.
#[inline]
pub fn is_sep_verbatim(c: char) -> bool {
    c == SEP
}

/// Returns whether the given byte is a path separator.
/// Allows both the primary separator '\' and the alternative separator '/'.
#[inline]
pub fn is_sep_byte(u: &u8) -> bool {
    *u == SEP_BYTE || *u == SEP2_BYTE
}

/// Returns whether the given byte is a path separator.
/// Only allows the primary separator '\'; use is_sep_byte to allow '/'.
#[inline]
pub fn is_sep_byte_verbatim(u: &u8) -> bool {
    *u == SEP_BYTE
}

/// Prefix types for Path
#[deriving(Eq, Clone)]
pub enum PathPrefix {
    /// Prefix `\\?\`, uint is the length of the following component
    VerbatimPrefix(uint),
    /// Prefix `\\?\UNC\`, uints are the lengths of the UNC components
    VerbatimUNCPrefix(uint, uint),
    /// Prefix `\\?\C:\` (for any alphabetic character)
    VerbatimDiskPrefix,
    /// Prefix `\\.\`, uint is the length of the following component
    DeviceNSPrefix(uint),
    /// UNC prefix `\\server\share`, uints are the lengths of the server/share
    UNCPrefix(uint, uint),
    /// Prefix `C:` for any alphabetic character
    DiskPrefix
}

fn parse_prefix<'a>(mut path: &'a str) -> Option<PathPrefix> {
    if path.starts_with("\\\\") {
        // \\
        path = path.slice_from(2);
        if path.starts_with("?\\") {
            // \\?\
            path = path.slice_from(2);
            if path.starts_with("UNC\\") {
                // \\?\UNC\server\share
                path = path.slice_from(4);
                let (idx_a, idx_b) = match parse_two_comps(path, is_sep_verbatim) {
                    Some(x) => x,
                    None => (path.len(), 0)
                };
                return Some(VerbatimUNCPrefix(idx_a, idx_b));
            } else {
                // \\?\path
                let idx = path.find('\\');
                if idx == Some(2) && path[1] == ':' as u8 {
                    let c = path[0];
                    if c.is_ascii() && ::char::is_alphabetic(c as char) {
                        // \\?\C:\ path
                        return Some(VerbatimDiskPrefix);
                    }
                }
                let idx = idx.unwrap_or(path.len());
                return Some(VerbatimPrefix(idx));
            }
        } else if path.starts_with(".\\") {
            // \\.\path
            path = path.slice_from(2);
            let idx = path.find('\\').unwrap_or(path.len());
            return Some(DeviceNSPrefix(idx));
        }
        match parse_two_comps(path, is_sep) {
            Some((idx_a, idx_b)) if idx_a > 0 && idx_b > 0 => {
                // \\server\share
                return Some(UNCPrefix(idx_a, idx_b));
            }
            _ => ()
        }
    } else if path.len() > 1 && path[1] == ':' as u8 {
        // C:
        let c = path[0];
        if c.is_ascii() && ::char::is_alphabetic(c as char) {
            return Some(DiskPrefix);
        }
    }
    return None;

    fn parse_two_comps<'a>(mut path: &'a str, f: |char| -> bool)
                       -> Option<(uint, uint)> {
        let idx_a = match path.find(|x| f(x)) {
            None => return None,
            Some(x) => x
        };
        path = path.slice_from(idx_a+1);
        let idx_b = path.find(f).unwrap_or(path.len());
        Some((idx_a, idx_b))
    }
}

// None result means the string didn't need normalizing
fn normalize_helper<'a>(s: &'a str, prefix: Option<PathPrefix>) -> (bool,Option<~[&'a str]>) {
    let f = if !prefix_is_verbatim(prefix) { is_sep } else { is_sep_verbatim };
    let is_abs = s.len() > prefix_len(prefix) && f(s.char_at(prefix_len(prefix)));
    let s_ = s.slice_from(prefix_len(prefix));
    let s_ = if is_abs { s_.slice_from(1) } else { s_ };

    if is_abs && s_.is_empty() {
        return (is_abs, match prefix {
            Some(DiskPrefix) | None => (if is_sep_verbatim(s.char_at(prefix_len(prefix))) { None }
                                        else { Some(~[]) }),
            Some(_) => Some(~[]), // need to trim the trailing separator
        });
    }
    let mut comps: ~[&'a str] = ~[];
    let mut n_up = 0u;
    let mut changed = false;
    for comp in s_.split(f) {
        if comp.is_empty() { changed = true }
        else if comp == "." { changed = true }
        else if comp == ".." {
            let has_abs_prefix = match prefix {
                Some(DiskPrefix) => false,
                Some(_) => true,
                None => false
            };
            if (is_abs || has_abs_prefix) && comps.is_empty() { changed = true }
            else if comps.len() == n_up { comps.push(".."); n_up += 1 }
            else { comps.pop().unwrap(); changed = true }
        } else { comps.push(comp) }
    }
    if !changed && !prefix_is_verbatim(prefix) {
        changed = s.find(is_sep).is_some();
    }
    if changed {
        if comps.is_empty() && !is_abs && prefix.is_none() {
            if s == "." {
                return (is_abs, None);
            }
            comps.push(".");
        }
        (is_abs, Some(comps))
    } else {
        (is_abs, None)
    }
}

fn prefix_is_verbatim(p: Option<PathPrefix>) -> bool {
    match p {
        Some(VerbatimPrefix(_)) | Some(VerbatimUNCPrefix(_,_)) | Some(VerbatimDiskPrefix) => true,
        Some(DeviceNSPrefix(_)) => true, // not really sure, but I think so
        _ => false
    }
}

fn prefix_len(p: Option<PathPrefix>) -> uint {
    match p {
        None => 0,
        Some(VerbatimPrefix(x)) => 4 + x,
        Some(VerbatimUNCPrefix(x,y)) => 8 + x + 1 + y,
        Some(VerbatimDiskPrefix) => 6,
        Some(UNCPrefix(x,y)) => 2 + x + 1 + y,
        Some(DeviceNSPrefix(x)) => 4 + x,
        Some(DiskPrefix) => 2
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use super::*;
    use super::parse_prefix;

    macro_rules! t(
        (s: $path:expr, $exp:expr) => (
            {
                let path = $path;
                assert!(path.as_str() == Some($exp));
            }
        );
        (v: $path:expr, $exp:expr) => (
            {
                let path = $path;
                assert!(path.as_vec() == $exp);
            }
        )
    )

    macro_rules! b(
        ($($arg:expr),+) => (
            {
                static the_bytes: &'static [u8] = bytes!($($arg),+);
                the_bytes
            }
        )
    )

    #[test]
    fn test_parse_prefix() {
        macro_rules! t(
            ($path:expr, $exp:expr) => (
                {
                    let path = $path;
                    let exp = $exp;
                    let res = parse_prefix(path);
                    assert!(res == exp,
                            "parse_prefix(\"{}\"): expected {:?}, found {:?}", path, exp, res);
                }
            )
        )

        t!("\\\\SERVER\\share\\foo", Some(UNCPrefix(6,5)));
        t!("\\\\", None);
        t!("\\\\SERVER", None);
        t!("\\\\SERVER\\", None);
        t!("\\\\SERVER\\\\", None);
        t!("\\\\SERVER\\\\foo", None);
        t!("\\\\SERVER\\share", Some(UNCPrefix(6,5)));
        t!("\\\\SERVER/share/foo", Some(UNCPrefix(6,5)));
        t!("\\\\SERVER\\share/foo", Some(UNCPrefix(6,5)));
        t!("//SERVER/share/foo", None);
        t!("\\\\\\a\\b\\c", None);
        t!("\\\\?\\a\\b\\c", Some(VerbatimPrefix(1)));
        t!("\\\\?\\a/b/c", Some(VerbatimPrefix(5)));
        t!("//?/a/b/c", None);
        t!("\\\\.\\a\\b", Some(DeviceNSPrefix(1)));
        t!("\\\\.\\a/b", Some(DeviceNSPrefix(3)));
        t!("//./a/b", None);
        t!("\\\\?\\UNC\\server\\share\\foo", Some(VerbatimUNCPrefix(6,5)));
        t!("\\\\?\\UNC\\\\share\\foo", Some(VerbatimUNCPrefix(0,5)));
        t!("\\\\?\\UNC\\", Some(VerbatimUNCPrefix(0,0)));
        t!("\\\\?\\UNC\\server/share/foo", Some(VerbatimUNCPrefix(16,0)));
        t!("\\\\?\\UNC\\server", Some(VerbatimUNCPrefix(6,0)));
        t!("\\\\?\\UNC\\server\\", Some(VerbatimUNCPrefix(6,0)));
        t!("\\\\?\\UNC/server/share", Some(VerbatimPrefix(16)));
        t!("\\\\?\\UNC", Some(VerbatimPrefix(3)));
        t!("\\\\?\\C:\\a\\b.txt", Some(VerbatimDiskPrefix));
        t!("\\\\?\\z:\\", Some(VerbatimDiskPrefix));
        t!("\\\\?\\C:", Some(VerbatimPrefix(2)));
        t!("\\\\?\\C:a.txt", Some(VerbatimPrefix(7)));
        t!("\\\\?\\C:a\\b.txt", Some(VerbatimPrefix(3)));
        t!("\\\\?\\C:/a", Some(VerbatimPrefix(4)));
        t!("C:\\foo", Some(DiskPrefix));
        t!("z:/foo", Some(DiskPrefix));
        t!("d:", Some(DiskPrefix));
        t!("ab:", None);
        t!("ü:\\foo", None);
        t!("3:\\foo", None);
        t!(" :\\foo", None);
        t!("::\\foo", None);
        t!("\\\\?\\C:", Some(VerbatimPrefix(2)));
        t!("\\\\?\\z:\\", Some(VerbatimDiskPrefix));
        t!("\\\\?\\ab:\\", Some(VerbatimPrefix(3)));
        t!("\\\\?\\C:\\a", Some(VerbatimDiskPrefix));
        t!("\\\\?\\C:/a", Some(VerbatimPrefix(4)));
        t!("\\\\?\\C:\\a/b", Some(VerbatimDiskPrefix));
    }

    #[test]
    fn test_paths() {
        let empty: &[u8] = [];
        t!(v: Path::new(empty), b!("."));
        t!(v: Path::new(b!("\\")), b!("\\"));
        t!(v: Path::new(b!("a\\b\\c")), b!("a\\b\\c"));

        t!(s: Path::new(""), ".");
        t!(s: Path::new("\\"), "\\");
        t!(s: Path::new("hi"), "hi");
        t!(s: Path::new("hi\\"), "hi");
        t!(s: Path::new("\\lib"), "\\lib");
        t!(s: Path::new("\\lib\\"), "\\lib");
        t!(s: Path::new("hi\\there"), "hi\\there");
        t!(s: Path::new("hi\\there.txt"), "hi\\there.txt");
        t!(s: Path::new("/"), "\\");
        t!(s: Path::new("hi/"), "hi");
        t!(s: Path::new("/lib"), "\\lib");
        t!(s: Path::new("/lib/"), "\\lib");
        t!(s: Path::new("hi/there"), "hi\\there");

        t!(s: Path::new("hi\\there\\"), "hi\\there");
        t!(s: Path::new("hi\\..\\there"), "there");
        t!(s: Path::new("hi/../there"), "there");
        t!(s: Path::new("..\\hi\\there"), "..\\hi\\there");
        t!(s: Path::new("\\..\\hi\\there"), "\\hi\\there");
        t!(s: Path::new("/../hi/there"), "\\hi\\there");
        t!(s: Path::new("foo\\.."), ".");
        t!(s: Path::new("\\foo\\.."), "\\");
        t!(s: Path::new("\\foo\\..\\.."), "\\");
        t!(s: Path::new("\\foo\\..\\..\\bar"), "\\bar");
        t!(s: Path::new("\\.\\hi\\.\\there\\."), "\\hi\\there");
        t!(s: Path::new("\\.\\hi\\.\\there\\.\\.."), "\\hi");
        t!(s: Path::new("foo\\..\\.."), "..");
        t!(s: Path::new("foo\\..\\..\\.."), "..\\..");
        t!(s: Path::new("foo\\..\\..\\bar"), "..\\bar");

        assert_eq!(Path::new(b!("foo\\bar")).into_vec(), b!("foo\\bar").to_owned());
        assert_eq!(Path::new(b!("\\foo\\..\\..\\bar")).into_vec(),
                   b!("\\bar").to_owned());

        t!(s: Path::new("\\\\a"), "\\a");
        t!(s: Path::new("\\\\a\\"), "\\a");
        t!(s: Path::new("\\\\a\\b"), "\\\\a\\b");
        t!(s: Path::new("\\\\a\\b\\"), "\\\\a\\b");
        t!(s: Path::new("\\\\a\\b/"), "\\\\a\\b");
        t!(s: Path::new("\\\\\\b"), "\\b");
        t!(s: Path::new("\\\\a\\\\b"), "\\a\\b");
        t!(s: Path::new("\\\\a\\b\\c"), "\\\\a\\b\\c");
        t!(s: Path::new("\\\\server\\share/path"), "\\\\server\\share\\path");
        t!(s: Path::new("\\\\server/share/path"), "\\\\server\\share\\path");
        t!(s: Path::new("C:a\\b.txt"), "C:a\\b.txt");
        t!(s: Path::new("C:a/b.txt"), "C:a\\b.txt");
        t!(s: Path::new("z:\\a\\b.txt"), "Z:\\a\\b.txt");
        t!(s: Path::new("z:/a/b.txt"), "Z:\\a\\b.txt");
        t!(s: Path::new("ab:/a/b.txt"), "ab:\\a\\b.txt");
        t!(s: Path::new("C:\\"), "C:\\");
        t!(s: Path::new("C:"), "C:");
        t!(s: Path::new("q:"), "Q:");
        t!(s: Path::new("C:/"), "C:\\");
        t!(s: Path::new("C:\\foo\\.."), "C:\\");
        t!(s: Path::new("C:foo\\.."), "C:");
        t!(s: Path::new("C:\\a\\"), "C:\\a");
        t!(s: Path::new("C:\\a/"), "C:\\a");
        t!(s: Path::new("C:\\a\\b\\"), "C:\\a\\b");
        t!(s: Path::new("C:\\a\\b/"), "C:\\a\\b");
        t!(s: Path::new("C:a\\"), "C:a");
        t!(s: Path::new("C:a/"), "C:a");
        t!(s: Path::new("C:a\\b\\"), "C:a\\b");
        t!(s: Path::new("C:a\\b/"), "C:a\\b");
        t!(s: Path::new("\\\\?\\z:\\a\\b.txt"), "\\\\?\\z:\\a\\b.txt");
        t!(s: Path::new("\\\\?\\C:/a/b.txt"), "\\\\?\\C:/a/b.txt");
        t!(s: Path::new("\\\\?\\C:\\a/b.txt"), "\\\\?\\C:\\a/b.txt");
        t!(s: Path::new("\\\\?\\test\\a\\b.txt"), "\\\\?\\test\\a\\b.txt");
        t!(s: Path::new("\\\\?\\foo\\bar\\"), "\\\\?\\foo\\bar\\");
        t!(s: Path::new("\\\\.\\foo\\bar"), "\\\\.\\foo\\bar");
        t!(s: Path::new("\\\\.\\"), "\\\\.\\");
        t!(s: Path::new("\\\\?\\UNC\\server\\share\\foo"), "\\\\?\\UNC\\server\\share\\foo");
        t!(s: Path::new("\\\\?\\UNC\\server/share"), "\\\\?\\UNC\\server/share\\");
        t!(s: Path::new("\\\\?\\UNC\\server"), "\\\\?\\UNC\\server\\");
        t!(s: Path::new("\\\\?\\UNC\\"), "\\\\?\\UNC\\\\");
        t!(s: Path::new("\\\\?\\UNC"), "\\\\?\\UNC");

        // I'm not sure whether \\.\foo/bar should normalize to \\.\foo\bar
        // as information is sparse and this isn't really googleable.
        // I'm going to err on the side of not normalizing it, as this skips the filesystem
        t!(s: Path::new("\\\\.\\foo/bar"), "\\\\.\\foo/bar");
        t!(s: Path::new("\\\\.\\foo\\bar"), "\\\\.\\foo\\bar");
    }

    #[test]
    fn test_opt_paths() {
        assert!(Path::new_opt(b!("foo\\bar", 0)) == None);
        assert!(Path::new_opt(b!("foo\\bar", 0x80)) == None);
        t!(v: Path::new_opt(b!("foo\\bar")).unwrap(), b!("foo\\bar"));
        assert!(Path::new_opt("foo\\bar\0") == None);
        t!(s: Path::new_opt("foo\\bar").unwrap(), "foo\\bar");
    }

    #[test]
    fn test_null_byte() {
        use task;
        let result = task::try(proc() {
            Path::new(b!("foo/bar", 0))
        });
        assert!(result.is_err());

        let result = task::try(proc() {
            Path::new("test").set_filename(b!("f", 0, "o"))
        });
        assert!(result.is_err());

        let result = task::try(proc() {
            Path::new("test").push(b!("f", 0, "o"));
        });
        assert!(result.is_err());
    }

    #[test]
    #[should_fail]
    fn test_not_utf8_fail() {
        Path::new(b!("hello", 0x80, ".txt"));
    }

    #[test]
    fn test_display_str() {
        let path = Path::new("foo");
        assert_eq!(path.display().to_str(), ~"foo");
        let path = Path::new(b!("\\"));
        assert_eq!(path.filename_display().to_str(), ~"");

        let path = Path::new("foo");
        let mo = path.display().as_maybe_owned();
        assert_eq!(mo.as_slice(), "foo");
        let path = Path::new(b!("\\"));
        let mo = path.filename_display().as_maybe_owned();
        assert_eq!(mo.as_slice(), "");
    }

    #[test]
    fn test_display() {
        macro_rules! t(
            ($path:expr, $exp:expr, $expf:expr) => (
                {
                    let path = Path::new($path);
                    let f = format!("{}", path.display());
                    assert_eq!(f.as_slice(), $exp);
                    let f = format!("{}", path.filename_display());
                    assert_eq!(f.as_slice(), $expf);
                }
            )
        )

        t!("foo", "foo", "foo");
        t!("foo\\bar", "foo\\bar", "bar");
        t!("\\", "\\", "");
    }

    #[test]
    fn test_components() {
        macro_rules! t(
            (s: $path:expr, $op:ident, $exp:expr) => (
                {
                    let path = $path;
                    let path = Path::new(path);
                    assert!(path.$op() == Some($exp));
                }
            );
            (s: $path:expr, $op:ident, $exp:expr, opt) => (
                {
                    let path = $path;
                    let path = Path::new(path);
                    let left = path.$op();
                    assert!(left == $exp);
                }
            );
            (v: $path:expr, $op:ident, $exp:expr) => (
                {
                    let path = $path;
                    let path = Path::new(path);
                    assert!(path.$op() == $exp);
                }
            )
        )

        t!(v: b!("a\\b\\c"), filename, Some(b!("c")));
        t!(s: "a\\b\\c", filename_str, "c");
        t!(s: "\\a\\b\\c", filename_str, "c");
        t!(s: "a", filename_str, "a");
        t!(s: "\\a", filename_str, "a");
        t!(s: ".", filename_str, None, opt);
        t!(s: "\\", filename_str, None, opt);
        t!(s: "..", filename_str, None, opt);
        t!(s: "..\\..", filename_str, None, opt);
        t!(s: "c:\\foo.txt", filename_str, "foo.txt");
        t!(s: "C:\\", filename_str, None, opt);
        t!(s: "C:", filename_str, None, opt);
        t!(s: "\\\\server\\share\\foo.txt", filename_str, "foo.txt");
        t!(s: "\\\\server\\share", filename_str, None, opt);
        t!(s: "\\\\server", filename_str, "server");
        t!(s: "\\\\?\\bar\\foo.txt", filename_str, "foo.txt");
        t!(s: "\\\\?\\bar", filename_str, None, opt);
        t!(s: "\\\\?\\", filename_str, None, opt);
        t!(s: "\\\\?\\UNC\\server\\share\\foo.txt", filename_str, "foo.txt");
        t!(s: "\\\\?\\UNC\\server", filename_str, None, opt);
        t!(s: "\\\\?\\UNC\\", filename_str, None, opt);
        t!(s: "\\\\?\\C:\\foo.txt", filename_str, "foo.txt");
        t!(s: "\\\\?\\C:\\", filename_str, None, opt);
        t!(s: "\\\\?\\C:", filename_str, None, opt);
        t!(s: "\\\\?\\foo/bar", filename_str, None, opt);
        t!(s: "\\\\?\\C:/foo", filename_str, None, opt);
        t!(s: "\\\\.\\foo\\bar", filename_str, "bar");
        t!(s: "\\\\.\\foo", filename_str, None, opt);
        t!(s: "\\\\.\\foo/bar", filename_str, None, opt);
        t!(s: "\\\\.\\foo\\bar/baz", filename_str, "bar/baz");
        t!(s: "\\\\.\\", filename_str, None, opt);
        t!(s: "\\\\?\\a\\b\\", filename_str, "b");

        t!(v: b!("a\\b\\c"), dirname, b!("a\\b"));
        t!(s: "a\\b\\c", dirname_str, "a\\b");
        t!(s: "\\a\\b\\c", dirname_str, "\\a\\b");
        t!(s: "a", dirname_str, ".");
        t!(s: "\\a", dirname_str, "\\");
        t!(s: ".", dirname_str, ".");
        t!(s: "\\", dirname_str, "\\");
        t!(s: "..", dirname_str, "..");
        t!(s: "..\\..", dirname_str, "..\\..");
        t!(s: "c:\\foo.txt", dirname_str, "C:\\");
        t!(s: "C:\\", dirname_str, "C:\\");
        t!(s: "C:", dirname_str, "C:");
        t!(s: "C:foo.txt", dirname_str, "C:");
        t!(s: "\\\\server\\share\\foo.txt", dirname_str, "\\\\server\\share");
        t!(s: "\\\\server\\share", dirname_str, "\\\\server\\share");
        t!(s: "\\\\server", dirname_str, "\\");
        t!(s: "\\\\?\\bar\\foo.txt", dirname_str, "\\\\?\\bar");
        t!(s: "\\\\?\\bar", dirname_str, "\\\\?\\bar");
        t!(s: "\\\\?\\", dirname_str, "\\\\?\\");
        t!(s: "\\\\?\\UNC\\server\\share\\foo.txt", dirname_str, "\\\\?\\UNC\\server\\share");
        t!(s: "\\\\?\\UNC\\server", dirname_str, "\\\\?\\UNC\\server\\");
        t!(s: "\\\\?\\UNC\\", dirname_str, "\\\\?\\UNC\\\\");
        t!(s: "\\\\?\\C:\\foo.txt", dirname_str, "\\\\?\\C:\\");
        t!(s: "\\\\?\\C:\\", dirname_str, "\\\\?\\C:\\");
        t!(s: "\\\\?\\C:", dirname_str, "\\\\?\\C:");
        t!(s: "\\\\?\\C:/foo/bar", dirname_str, "\\\\?\\C:/foo/bar");
        t!(s: "\\\\?\\foo/bar", dirname_str, "\\\\?\\foo/bar");
        t!(s: "\\\\.\\foo\\bar", dirname_str, "\\\\.\\foo");
        t!(s: "\\\\.\\foo", dirname_str, "\\\\.\\foo");
        t!(s: "\\\\?\\a\\b\\", dirname_str, "\\\\?\\a");

        t!(v: b!("hi\\there.txt"), filestem, Some(b!("there")));
        t!(s: "hi\\there.txt", filestem_str, "there");
        t!(s: "hi\\there", filestem_str, "there");
        t!(s: "there.txt", filestem_str, "there");
        t!(s: "there", filestem_str, "there");
        t!(s: ".", filestem_str, None, opt);
        t!(s: "\\", filestem_str, None, opt);
        t!(s: "foo\\.bar", filestem_str, ".bar");
        t!(s: ".bar", filestem_str, ".bar");
        t!(s: "..bar", filestem_str, ".");
        t!(s: "hi\\there..txt", filestem_str, "there.");
        t!(s: "..", filestem_str, None, opt);
        t!(s: "..\\..", filestem_str, None, opt);
        // filestem is based on filename, so we don't need the full set of prefix tests

        t!(v: b!("hi\\there.txt"), extension, Some(b!("txt")));
        t!(v: b!("hi\\there"), extension, None);
        t!(s: "hi\\there.txt", extension_str, Some("txt"), opt);
        t!(s: "hi\\there", extension_str, None, opt);
        t!(s: "there.txt", extension_str, Some("txt"), opt);
        t!(s: "there", extension_str, None, opt);
        t!(s: ".", extension_str, None, opt);
        t!(s: "\\", extension_str, None, opt);
        t!(s: "foo\\.bar", extension_str, None, opt);
        t!(s: ".bar", extension_str, None, opt);
        t!(s: "..bar", extension_str, Some("bar"), opt);
        t!(s: "hi\\there..txt", extension_str, Some("txt"), opt);
        t!(s: "..", extension_str, None, opt);
        t!(s: "..\\..", extension_str, None, opt);
        // extension is based on filename, so we don't need the full set of prefix tests
    }

    #[test]
    fn test_push() {
        macro_rules! t(
            (s: $path:expr, $join:expr) => (
                {
                    let path = $path;
                    let join = $join;
                    let mut p1 = Path::new(path);
                    let p2 = p1.clone();
                    p1.push(join);
                    assert!(p1 == p2.join(join));
                }
            )
        )

        t!(s: "a\\b\\c", "..");
        t!(s: "\\a\\b\\c", "d");
        t!(s: "a\\b", "c\\d");
        t!(s: "a\\b", "\\c\\d");
        // this is just a sanity-check test. push and join share an implementation,
        // so there's no need for the full set of prefix tests

        // we do want to check one odd case though to ensure the prefix is re-parsed
        let mut p = Path::new("\\\\?\\C:");
        assert!(prefix(&p) == Some(VerbatimPrefix(2)));
        p.push("foo");
        assert!(prefix(&p) == Some(VerbatimDiskPrefix));
        assert_eq!(p.as_str(), Some("\\\\?\\C:\\foo"));

        // and another with verbatim non-normalized paths
        let mut p = Path::new("\\\\?\\C:\\a\\");
        p.push("foo");
        assert_eq!(p.as_str(), Some("\\\\?\\C:\\a\\foo"));
    }

    #[test]
    fn test_push_path() {
        macro_rules! t(
            (s: $path:expr, $push:expr, $exp:expr) => (
                {
                    let mut p = Path::new($path);
                    let push = Path::new($push);
                    p.push(&push);
                    assert_eq!(p.as_str(), Some($exp));
                }
            )
        )

        t!(s: "a\\b\\c", "d", "a\\b\\c\\d");
        t!(s: "\\a\\b\\c", "d", "\\a\\b\\c\\d");
        t!(s: "a\\b", "c\\d", "a\\b\\c\\d");
        t!(s: "a\\b", "\\c\\d", "\\c\\d");
        t!(s: "a\\b", ".", "a\\b");
        t!(s: "a\\b", "..\\c", "a\\c");
        t!(s: "a\\b", "C:a.txt", "C:a.txt");
        t!(s: "a\\b", "..\\..\\..\\c", "..\\c");
        t!(s: "a\\b", "C:\\a.txt", "C:\\a.txt");
        t!(s: "C:\\a", "C:\\b.txt", "C:\\b.txt");
        t!(s: "C:\\a\\b\\c", "C:d", "C:\\a\\b\\c\\d");
        t!(s: "C:a\\b\\c", "C:d", "C:a\\b\\c\\d");
        t!(s: "C:a\\b", "..\\..\\..\\c", "C:..\\c");
        t!(s: "C:\\a\\b", "..\\..\\..\\c", "C:\\c");
        t!(s: "C:", r"a\b\c", r"C:a\b\c");
        t!(s: "C:", r"..\a", r"C:..\a");
        t!(s: "\\\\server\\share\\foo", "bar", "\\\\server\\share\\foo\\bar");
        t!(s: "\\\\server\\share\\foo", "..\\..\\bar", "\\\\server\\share\\bar");
        t!(s: "\\\\server\\share\\foo", "C:baz", "C:baz");
        t!(s: "\\\\?\\C:\\a\\b", "C:c\\d", "\\\\?\\C:\\a\\b\\c\\d");
        t!(s: "\\\\?\\C:a\\b", "C:c\\d", "C:c\\d");
        t!(s: "\\\\?\\C:\\a\\b", "C:\\c\\d", "C:\\c\\d");
        t!(s: "\\\\?\\foo\\bar", "baz", "\\\\?\\foo\\bar\\baz");
        t!(s: "\\\\?\\C:\\a\\b", "..\\..\\..\\c", "\\\\?\\C:\\a\\b\\..\\..\\..\\c");
        t!(s: "\\\\?\\foo\\bar", "..\\..\\c", "\\\\?\\foo\\bar\\..\\..\\c");
        t!(s: "\\\\?\\", "foo", "\\\\?\\\\foo");
        t!(s: "\\\\?\\UNC\\server\\share\\foo", "bar", "\\\\?\\UNC\\server\\share\\foo\\bar");
        t!(s: "\\\\?\\UNC\\server\\share", "C:\\a", "C:\\a");
        t!(s: "\\\\?\\UNC\\server\\share", "C:a", "C:a");
        t!(s: "\\\\?\\UNC\\server", "foo", "\\\\?\\UNC\\server\\\\foo");
        t!(s: "C:\\a", "\\\\?\\UNC\\server\\share", "\\\\?\\UNC\\server\\share");
        t!(s: "\\\\.\\foo\\bar", "baz", "\\\\.\\foo\\bar\\baz");
        t!(s: "\\\\.\\foo\\bar", "C:a", "C:a");
        // again, not sure about the following, but I'm assuming \\.\ should be verbatim
        t!(s: "\\\\.\\foo", "..\\bar", "\\\\.\\foo\\..\\bar");

        t!(s: "\\\\?\\C:", "foo", "\\\\?\\C:\\foo"); // this is a weird one
    }

    #[test]
    fn test_push_many() {
        macro_rules! t(
            (s: $path:expr, $push:expr, $exp:expr) => (
                {
                    let mut p = Path::new($path);
                    p.push_many($push);
                    assert_eq!(p.as_str(), Some($exp));
                }
            );
            (v: $path:expr, $push:expr, $exp:expr) => (
                {
                    let mut p = Path::new($path);
                    p.push_many($push);
                    assert_eq!(p.as_vec(), $exp);
                }
            )
        )

        t!(s: "a\\b\\c", ["d", "e"], "a\\b\\c\\d\\e");
        t!(s: "a\\b\\c", ["d", "\\e"], "\\e");
        t!(s: "a\\b\\c", ["d", "\\e", "f"], "\\e\\f");
        t!(s: "a\\b\\c", [~"d", ~"e"], "a\\b\\c\\d\\e");
        t!(v: b!("a\\b\\c"), [b!("d"), b!("e")], b!("a\\b\\c\\d\\e"));
        t!(v: b!("a\\b\\c"), [b!("d"), b!("\\e"), b!("f")], b!("\\e\\f"));
        t!(v: b!("a\\b\\c"), [b!("d").to_owned(), b!("e").to_owned()], b!("a\\b\\c\\d\\e"));
    }

    #[test]
    fn test_pop() {
        macro_rules! t(
            (s: $path:expr, $left:expr, $right:expr) => (
                {
                    let pstr = $path;
                    let mut p = Path::new(pstr);
                    let result = p.pop();
                    let left = $left;
                    assert!(p.as_str() == Some(left),
                        "`{}`.pop() failed; expected remainder `{}`, found `{}`",
                        pstr, left, p.as_str().unwrap());
                    assert!(result == $right);
                }
            );
            (v: [$($path:expr),+], [$($left:expr),+], $right:expr) => (
                {
                    let mut p = Path::new(b!($($path),+));
                    let result = p.pop();
                    assert_eq!(p.as_vec(), b!($($left),+));
                    assert!(result == $right);
                }
            )
        )

        t!(s: "a\\b\\c", "a\\b", true);
        t!(s: "a", ".", true);
        t!(s: ".", ".", false);
        t!(s: "\\a", "\\", true);
        t!(s: "\\", "\\", false);
        t!(v: ["a\\b\\c"], ["a\\b"], true);
        t!(v: ["a"], ["."], true);
        t!(v: ["."], ["."], false);
        t!(v: ["\\a"], ["\\"], true);
        t!(v: ["\\"], ["\\"], false);

        t!(s: "C:\\a\\b", "C:\\a", true);
        t!(s: "C:\\a", "C:\\", true);
        t!(s: "C:\\", "C:\\", false);
        t!(s: "C:a\\b", "C:a", true);
        t!(s: "C:a", "C:", true);
        t!(s: "C:", "C:", false);
        t!(s: "\\\\server\\share\\a\\b", "\\\\server\\share\\a", true);
        t!(s: "\\\\server\\share\\a", "\\\\server\\share", true);
        t!(s: "\\\\server\\share", "\\\\server\\share", false);
        t!(s: "\\\\?\\a\\b\\c", "\\\\?\\a\\b", true);
        t!(s: "\\\\?\\a\\b", "\\\\?\\a", true);
        t!(s: "\\\\?\\a", "\\\\?\\a", false);
        t!(s: "\\\\?\\C:\\a\\b", "\\\\?\\C:\\a", true);
        t!(s: "\\\\?\\C:\\a", "\\\\?\\C:\\", true);
        t!(s: "\\\\?\\C:\\", "\\\\?\\C:\\", false);
        t!(s: "\\\\?\\UNC\\server\\share\\a\\b", "\\\\?\\UNC\\server\\share\\a", true);
        t!(s: "\\\\?\\UNC\\server\\share\\a", "\\\\?\\UNC\\server\\share", true);
        t!(s: "\\\\?\\UNC\\server\\share", "\\\\?\\UNC\\server\\share", false);
        t!(s: "\\\\.\\a\\b\\c", "\\\\.\\a\\b", true);
        t!(s: "\\\\.\\a\\b", "\\\\.\\a", true);
        t!(s: "\\\\.\\a", "\\\\.\\a", false);

        t!(s: "\\\\?\\a\\b\\", "\\\\?\\a", true);
    }

    #[test]
    fn test_root_path() {
        assert!(Path::new("a\\b\\c").root_path() == None);
        assert!(Path::new("\\a\\b\\c").root_path() == Some(Path::new("\\")));
        assert!(Path::new("C:a").root_path() == Some(Path::new("C:")));
        assert!(Path::new("C:\\a").root_path() == Some(Path::new("C:\\")));
        assert!(Path::new("\\\\a\\b\\c").root_path() == Some(Path::new("\\\\a\\b")));
        assert!(Path::new("\\\\?\\a\\b").root_path() == Some(Path::new("\\\\?\\a")));
        assert!(Path::new("\\\\?\\C:\\a").root_path() == Some(Path::new("\\\\?\\C:\\")));
        assert!(Path::new("\\\\?\\UNC\\a\\b\\c").root_path() ==
                Some(Path::new("\\\\?\\UNC\\a\\b")));
        assert!(Path::new("\\\\.\\a\\b").root_path() == Some(Path::new("\\\\.\\a")));
    }

    #[test]
    fn test_join() {
        t!(s: Path::new("a\\b\\c").join(".."), "a\\b");
        t!(s: Path::new("\\a\\b\\c").join("d"), "\\a\\b\\c\\d");
        t!(s: Path::new("a\\b").join("c\\d"), "a\\b\\c\\d");
        t!(s: Path::new("a\\b").join("\\c\\d"), "\\c\\d");
        t!(s: Path::new(".").join("a\\b"), "a\\b");
        t!(s: Path::new("\\").join("a\\b"), "\\a\\b");
        t!(v: Path::new(b!("a\\b\\c")).join(b!("..")), b!("a\\b"));
        t!(v: Path::new(b!("\\a\\b\\c")).join(b!("d")), b!("\\a\\b\\c\\d"));
        // full join testing is covered under test_push_path, so no need for
        // the full set of prefix tests
    }

    #[test]
    fn test_join_path() {
        macro_rules! t(
            (s: $path:expr, $join:expr, $exp:expr) => (
                {
                    let path = Path::new($path);
                    let join = Path::new($join);
                    let res = path.join(&join);
                    assert_eq!(res.as_str(), Some($exp));
                }
            )
        )

        t!(s: "a\\b\\c", "..", "a\\b");
        t!(s: "\\a\\b\\c", "d", "\\a\\b\\c\\d");
        t!(s: "a\\b", "c\\d", "a\\b\\c\\d");
        t!(s: "a\\b", "\\c\\d", "\\c\\d");
        t!(s: ".", "a\\b", "a\\b");
        t!(s: "\\", "a\\b", "\\a\\b");
        // join is implemented using push, so there's no need for
        // the full set of prefix tests
    }

    #[test]
    fn test_join_many() {
        macro_rules! t(
            (s: $path:expr, $join:expr, $exp:expr) => (
                {
                    let path = Path::new($path);
                    let res = path.join_many($join);
                    assert_eq!(res.as_str(), Some($exp));
                }
            );
            (v: $path:expr, $join:expr, $exp:expr) => (
                {
                    let path = Path::new($path);
                    let res = path.join_many($join);
                    assert_eq!(res.as_vec(), $exp);
                }
            )
        )

        t!(s: "a\\b\\c", ["d", "e"], "a\\b\\c\\d\\e");
        t!(s: "a\\b\\c", ["..", "d"], "a\\b\\d");
        t!(s: "a\\b\\c", ["d", "\\e", "f"], "\\e\\f");
        t!(s: "a\\b\\c", [~"d", ~"e"], "a\\b\\c\\d\\e");
        t!(v: b!("a\\b\\c"), [b!("d"), b!("e")], b!("a\\b\\c\\d\\e"));
        t!(v: b!("a\\b\\c"), [b!("d").to_owned(), b!("e").to_owned()], b!("a\\b\\c\\d\\e"));
    }

    #[test]
    fn test_with_helpers() {
        macro_rules! t(
            (s: $path:expr, $op:ident, $arg:expr, $res:expr) => (
                {
                    let pstr = $path;
                    let path = Path::new(pstr);
                    let arg = $arg;
                    let res = path.$op(arg);
                    let exp = $res;
                    assert!(res.as_str() == Some(exp),
                            "`{}`.{}(\"{}\"): Expected `{}`, found `{}`",
                            pstr, stringify!($op), arg, exp, res.as_str().unwrap());
                }
            )
        )

        t!(s: "a\\b\\c", with_filename, "d", "a\\b\\d");
        t!(s: ".", with_filename, "foo", "foo");
        t!(s: "\\a\\b\\c", with_filename, "d", "\\a\\b\\d");
        t!(s: "\\", with_filename, "foo", "\\foo");
        t!(s: "\\a", with_filename, "foo", "\\foo");
        t!(s: "foo", with_filename, "bar", "bar");
        t!(s: "\\", with_filename, "foo\\", "\\foo");
        t!(s: "\\a", with_filename, "foo\\", "\\foo");
        t!(s: "a\\b\\c", with_filename, "", "a\\b");
        t!(s: "a\\b\\c", with_filename, ".", "a\\b");
        t!(s: "a\\b\\c", with_filename, "..", "a");
        t!(s: "\\a", with_filename, "", "\\");
        t!(s: "foo", with_filename, "", ".");
        t!(s: "a\\b\\c", with_filename, "d\\e", "a\\b\\d\\e");
        t!(s: "a\\b\\c", with_filename, "\\d", "a\\b\\d");
        t!(s: "..", with_filename, "foo", "..\\foo");
        t!(s: "..\\..", with_filename, "foo", "..\\..\\foo");
        t!(s: "..", with_filename, "", "..");
        t!(s: "..\\..", with_filename, "", "..\\..");
        t!(s: "C:\\foo\\bar", with_filename, "baz", "C:\\foo\\baz");
        t!(s: "C:\\foo", with_filename, "bar", "C:\\bar");
        t!(s: "C:\\", with_filename, "foo", "C:\\foo");
        t!(s: "C:foo\\bar", with_filename, "baz", "C:foo\\baz");
        t!(s: "C:foo", with_filename, "bar", "C:bar");
        t!(s: "C:", with_filename, "foo", "C:foo");
        t!(s: "C:\\foo", with_filename, "", "C:\\");
        t!(s: "C:foo", with_filename, "", "C:");
        t!(s: "C:\\foo\\bar", with_filename, "..", "C:\\");
        t!(s: "C:\\foo", with_filename, "..", "C:\\");
        t!(s: "C:\\", with_filename, "..", "C:\\");
        t!(s: "C:foo\\bar", with_filename, "..", "C:");
        t!(s: "C:foo", with_filename, "..", "C:..");
        t!(s: "C:", with_filename, "..", "C:..");
        t!(s: "\\\\server\\share\\foo", with_filename, "bar", "\\\\server\\share\\bar");
        t!(s: "\\\\server\\share", with_filename, "foo", "\\\\server\\share\\foo");
        t!(s: "\\\\server\\share\\foo", with_filename, "", "\\\\server\\share");
        t!(s: "\\\\server\\share", with_filename, "", "\\\\server\\share");
        t!(s: "\\\\server\\share\\foo", with_filename, "..", "\\\\server\\share");
        t!(s: "\\\\server\\share", with_filename, "..", "\\\\server\\share");
        t!(s: "\\\\?\\C:\\foo\\bar", with_filename, "baz", "\\\\?\\C:\\foo\\baz");
        t!(s: "\\\\?\\C:\\foo", with_filename, "bar", "\\\\?\\C:\\bar");
        t!(s: "\\\\?\\C:\\", with_filename, "foo", "\\\\?\\C:\\foo");
        t!(s: "\\\\?\\C:\\foo", with_filename, "..", "\\\\?\\C:\\..");
        t!(s: "\\\\?\\foo\\bar", with_filename, "baz", "\\\\?\\foo\\baz");
        t!(s: "\\\\?\\foo", with_filename, "bar", "\\\\?\\foo\\bar");
        t!(s: "\\\\?\\", with_filename, "foo", "\\\\?\\\\foo");
        t!(s: "\\\\?\\foo\\bar", with_filename, "..", "\\\\?\\foo\\..");
        t!(s: "\\\\.\\foo\\bar", with_filename, "baz", "\\\\.\\foo\\baz");
        t!(s: "\\\\.\\foo", with_filename, "bar", "\\\\.\\foo\\bar");
        t!(s: "\\\\.\\foo\\bar", with_filename, "..", "\\\\.\\foo\\..");

        t!(s: "hi\\there.txt", with_extension, "exe", "hi\\there.exe");
        t!(s: "hi\\there.txt", with_extension, "", "hi\\there");
        t!(s: "hi\\there.txt", with_extension, ".", "hi\\there..");
        t!(s: "hi\\there.txt", with_extension, "..", "hi\\there...");
        t!(s: "hi\\there", with_extension, "txt", "hi\\there.txt");
        t!(s: "hi\\there", with_extension, ".", "hi\\there..");
        t!(s: "hi\\there", with_extension, "..", "hi\\there...");
        t!(s: "hi\\there.", with_extension, "txt", "hi\\there.txt");
        t!(s: "hi\\.foo", with_extension, "txt", "hi\\.foo.txt");
        t!(s: "hi\\there.txt", with_extension, ".foo", "hi\\there..foo");
        t!(s: "\\", with_extension, "txt", "\\");
        t!(s: "\\", with_extension, ".", "\\");
        t!(s: "\\", with_extension, "..", "\\");
        t!(s: ".", with_extension, "txt", ".");
        // extension setter calls filename setter internally, no need for extended tests
    }

    #[test]
    fn test_setters() {
        macro_rules! t(
            (s: $path:expr, $set:ident, $with:ident, $arg:expr) => (
                {
                    let path = $path;
                    let arg = $arg;
                    let mut p1 = Path::new(path);
                    p1.$set(arg);
                    let p2 = Path::new(path);
                    assert!(p1 == p2.$with(arg));
                }
            );
            (v: $path:expr, $set:ident, $with:ident, $arg:expr) => (
                {
                    let path = $path;
                    let arg = $arg;
                    let mut p1 = Path::new(path);
                    p1.$set(arg);
                    let p2 = Path::new(path);
                    assert!(p1 == p2.$with(arg));
                }
            )
        )

        t!(v: b!("a\\b\\c"), set_filename, with_filename, b!("d"));
        t!(v: b!("\\"), set_filename, with_filename, b!("foo"));
        t!(s: "a\\b\\c", set_filename, with_filename, "d");
        t!(s: "\\", set_filename, with_filename, "foo");
        t!(s: ".", set_filename, with_filename, "foo");
        t!(s: "a\\b", set_filename, with_filename, "");
        t!(s: "a", set_filename, with_filename, "");

        t!(v: b!("hi\\there.txt"), set_extension, with_extension, b!("exe"));
        t!(s: "hi\\there.txt", set_extension, with_extension, "exe");
        t!(s: "hi\\there.", set_extension, with_extension, "txt");
        t!(s: "hi\\there", set_extension, with_extension, "txt");
        t!(s: "hi\\there.txt", set_extension, with_extension, "");
        t!(s: "hi\\there", set_extension, with_extension, "");
        t!(s: ".", set_extension, with_extension, "txt");

        // with_ helpers use the setter internally, so the tests for the with_ helpers
        // will suffice. No need for the full set of prefix tests.
    }

    #[test]
    fn test_getters() {
        macro_rules! t(
            (s: $path:expr, $filename:expr, $dirname:expr, $filestem:expr, $ext:expr) => (
                {
                    let path = $path;
                    let filename = $filename;
                    assert!(path.filename_str() == filename,
                            "`{}`.filename_str(): Expected `{:?}`, found `{:?}`",
                            path.as_str().unwrap(), filename, path.filename_str());
                    let dirname = $dirname;
                    assert!(path.dirname_str() == dirname,
                            "`{}`.dirname_str(): Expected `{:?}`, found `{:?}`",
                            path.as_str().unwrap(), dirname, path.dirname_str());
                    let filestem = $filestem;
                    assert!(path.filestem_str() == filestem,
                            "`{}`.filestem_str(): Expected `{:?}`, found `{:?}`",
                            path.as_str().unwrap(), filestem, path.filestem_str());
                    let ext = $ext;
                    assert!(path.extension_str() == ext,
                            "`{}`.extension_str(): Expected `{:?}`, found `{:?}`",
                            path.as_str().unwrap(), ext, path.extension_str());
                }
            );
            (v: $path:expr, $filename:expr, $dirname:expr, $filestem:expr, $ext:expr) => (
                {
                    let path = $path;
                    assert!(path.filename() == $filename);
                    assert!(path.dirname() == $dirname);
                    assert!(path.filestem() == $filestem);
                    assert!(path.extension() == $ext);
                }
            )
        )

        t!(v: Path::new(b!("a\\b\\c")), Some(b!("c")), b!("a\\b"), Some(b!("c")), None);
        t!(s: Path::new("a\\b\\c"), Some("c"), Some("a\\b"), Some("c"), None);
        t!(s: Path::new("."), None, Some("."), None, None);
        t!(s: Path::new("\\"), None, Some("\\"), None, None);
        t!(s: Path::new(".."), None, Some(".."), None, None);
        t!(s: Path::new("..\\.."), None, Some("..\\.."), None, None);
        t!(s: Path::new("hi\\there.txt"), Some("there.txt"), Some("hi"),
              Some("there"), Some("txt"));
        t!(s: Path::new("hi\\there"), Some("there"), Some("hi"), Some("there"), None);
        t!(s: Path::new("hi\\there."), Some("there."), Some("hi"),
              Some("there"), Some(""));
        t!(s: Path::new("hi\\.there"), Some(".there"), Some("hi"), Some(".there"), None);
        t!(s: Path::new("hi\\..there"), Some("..there"), Some("hi"),
              Some("."), Some("there"));

        // these are already tested in test_components, so no need for extended tests
    }

    #[test]
    fn test_dir_path() {
        t!(s: Path::new("hi\\there").dir_path(), "hi");
        t!(s: Path::new("hi").dir_path(), ".");
        t!(s: Path::new("\\hi").dir_path(), "\\");
        t!(s: Path::new("\\").dir_path(), "\\");
        t!(s: Path::new("..").dir_path(), "..");
        t!(s: Path::new("..\\..").dir_path(), "..\\..");

        // dir_path is just dirname interpreted as a path.
        // No need for extended tests
    }

    #[test]
    fn test_is_absolute() {
        macro_rules! t(
            ($path:expr, $abs:expr, $vol:expr, $cwd:expr, $rel:expr) => (
                {
                    let path = Path::new($path);
                    let (abs, vol, cwd, rel) = ($abs, $vol, $cwd, $rel);
                    let b = path.is_absolute();
                    assert!(b == abs, "Path '{}'.is_absolute(): expected {:?}, found {:?}",
                            path.as_str().unwrap(), abs, b);
                    let b = is_vol_relative(&path);
                    assert!(b == vol, "is_vol_relative('{}'): expected {:?}, found {:?}",
                            path.as_str().unwrap(), vol, b);
                    let b = is_cwd_relative(&path);
                    assert!(b == cwd, "is_cwd_relative('{}'): expected {:?}, found {:?}",
                            path.as_str().unwrap(), cwd, b);
                    let b = path.is_relative();
                    assert!(b == rel, "Path '{}'.is_relativf(): expected {:?}, found {:?}",
                            path.as_str().unwrap(), rel, b);
                }
            )
        )
        t!("a\\b\\c", false, false, false, true);
        t!("\\a\\b\\c", false, true, false, false);
        t!("a", false, false, false, true);
        t!("\\a", false, true, false, false);
        t!(".", false, false, false, true);
        t!("\\", false, true, false, false);
        t!("..", false, false, false, true);
        t!("..\\..", false, false, false, true);
        t!("C:a\\b.txt", false, false, true, false);
        t!("C:\\a\\b.txt", true, false, false, false);
        t!("\\\\server\\share\\a\\b.txt", true, false, false, false);
        t!("\\\\?\\a\\b\\c.txt", true, false, false, false);
        t!("\\\\?\\C:\\a\\b.txt", true, false, false, false);
        t!("\\\\?\\C:a\\b.txt", true, false, false, false); // NB: not equivalent to C:a\b.txt
        t!("\\\\?\\UNC\\server\\share\\a\\b.txt", true, false, false, false);
        t!("\\\\.\\a\\b", true, false, false, false);
    }

    #[test]
    fn test_is_ancestor_of() {
        macro_rules! t(
            (s: $path:expr, $dest:expr, $exp:expr) => (
                {
                    let path = Path::new($path);
                    let dest = Path::new($dest);
                    let exp = $exp;
                    let res = path.is_ancestor_of(&dest);
                    assert!(res == exp,
                            "`{}`.is_ancestor_of(`{}`): Expected {:?}, found {:?}",
                            path.as_str().unwrap(), dest.as_str().unwrap(), exp, res);
                }
            )
        )

        t!(s: "a\\b\\c", "a\\b\\c\\d", true);
        t!(s: "a\\b\\c", "a\\b\\c", true);
        t!(s: "a\\b\\c", "a\\b", false);
        t!(s: "\\a\\b\\c", "\\a\\b\\c", true);
        t!(s: "\\a\\b", "\\a\\b\\c", true);
        t!(s: "\\a\\b\\c\\d", "\\a\\b\\c", false);
        t!(s: "\\a\\b", "a\\b\\c", false);
        t!(s: "a\\b", "\\a\\b\\c", false);
        t!(s: "a\\b\\c", "a\\b\\d", false);
        t!(s: "..\\a\\b\\c", "a\\b\\c", false);
        t!(s: "a\\b\\c", "..\\a\\b\\c", false);
        t!(s: "a\\b\\c", "a\\b\\cd", false);
        t!(s: "a\\b\\cd", "a\\b\\c", false);
        t!(s: "..\\a\\b", "..\\a\\b\\c", true);
        t!(s: ".", "a\\b", true);
        t!(s: ".", ".", true);
        t!(s: "\\", "\\", true);
        t!(s: "\\", "\\a\\b", true);
        t!(s: "..", "a\\b", true);
        t!(s: "..\\..", "a\\b", true);
        t!(s: "foo\\bar", "foobar", false);
        t!(s: "foobar", "foo\\bar", false);

        t!(s: "foo", "C:foo", false);
        t!(s: "C:foo", "foo", false);
        t!(s: "C:foo", "C:foo\\bar", true);
        t!(s: "C:foo\\bar", "C:foo", false);
        t!(s: "C:\\foo", "C:\\foo\\bar", true);
        t!(s: "C:", "C:", true);
        t!(s: "C:", "C:\\", false);
        t!(s: "C:\\", "C:", false);
        t!(s: "C:\\", "C:\\", true);
        t!(s: "C:\\foo\\bar", "C:\\foo", false);
        t!(s: "C:foo\\bar", "C:foo", false);
        t!(s: "C:\\foo", "\\foo", false);
        t!(s: "\\foo", "C:\\foo", false);
        t!(s: "\\\\server\\share\\foo", "\\\\server\\share\\foo\\bar", true);
        t!(s: "\\\\server\\share", "\\\\server\\share\\foo", true);
        t!(s: "\\\\server\\share\\foo", "\\\\server\\share", false);
        t!(s: "C:\\foo", "\\\\server\\share\\foo", false);
        t!(s: "\\\\server\\share\\foo", "C:\\foo", false);
        t!(s: "\\\\?\\foo\\bar", "\\\\?\\foo\\bar\\baz", true);
        t!(s: "\\\\?\\foo\\bar\\baz", "\\\\?\\foo\\bar", false);
        t!(s: "\\\\?\\foo\\bar", "\\foo\\bar\\baz", false);
        t!(s: "\\foo\\bar", "\\\\?\\foo\\bar\\baz", false);
        t!(s: "\\\\?\\C:\\foo\\bar", "\\\\?\\C:\\foo\\bar\\baz", true);
        t!(s: "\\\\?\\C:\\foo\\bar\\baz", "\\\\?\\C:\\foo\\bar", false);
        t!(s: "\\\\?\\C:\\", "\\\\?\\C:\\foo", true);
        t!(s: "\\\\?\\C:", "\\\\?\\C:\\", false); // this is a weird one
        t!(s: "\\\\?\\C:\\", "\\\\?\\C:", false);
        t!(s: "\\\\?\\C:\\a", "\\\\?\\c:\\a\\b", true);
        t!(s: "\\\\?\\c:\\a", "\\\\?\\C:\\a\\b", true);
        t!(s: "\\\\?\\C:\\a", "\\\\?\\D:\\a\\b", false);
        t!(s: "\\\\?\\foo", "\\\\?\\foobar", false);
        t!(s: "\\\\?\\a\\b", "\\\\?\\a\\b\\c", true);
        t!(s: "\\\\?\\a\\b", "\\\\?\\a\\b\\", true);
        t!(s: "\\\\?\\a\\b\\", "\\\\?\\a\\b", true);
        t!(s: "\\\\?\\a\\b\\c", "\\\\?\\a\\b", false);
        t!(s: "\\\\?\\a\\b\\c", "\\\\?\\a\\b\\", false);
        t!(s: "\\\\?\\UNC\\a\\b\\c", "\\\\?\\UNC\\a\\b\\c\\d", true);
        t!(s: "\\\\?\\UNC\\a\\b\\c\\d", "\\\\?\\UNC\\a\\b\\c", false);
        t!(s: "\\\\?\\UNC\\a\\b", "\\\\?\\UNC\\a\\b\\c", true);
        t!(s: "\\\\.\\foo\\bar", "\\\\.\\foo\\bar\\baz", true);
        t!(s: "\\\\.\\foo\\bar\\baz", "\\\\.\\foo\\bar", false);
        t!(s: "\\\\.\\foo", "\\\\.\\foo\\bar", true);
        t!(s: "\\\\.\\foo", "\\\\.\\foobar", false);

        t!(s: "\\a\\b", "\\\\?\\a\\b", false);
        t!(s: "\\\\?\\a\\b", "\\a\\b", false);
        t!(s: "\\a\\b", "\\\\?\\C:\\a\\b", false);
        t!(s: "\\\\?\\C:\\a\\b", "\\a\\b", false);
        t!(s: "Z:\\a\\b", "\\\\?\\z:\\a\\b", true);
        t!(s: "C:\\a\\b", "\\\\?\\D:\\a\\b", false);
        t!(s: "a\\b", "\\\\?\\a\\b", false);
        t!(s: "\\\\?\\a\\b", "a\\b", false);
        t!(s: "C:\\a\\b", "\\\\?\\C:\\a\\b", true);
        t!(s: "\\\\?\\C:\\a\\b", "C:\\a\\b", true);
        t!(s: "C:a\\b", "\\\\?\\C:\\a\\b", false);
        t!(s: "C:a\\b", "\\\\?\\C:a\\b", false);
        t!(s: "\\\\?\\C:\\a\\b", "C:a\\b", false);
        t!(s: "\\\\?\\C:a\\b", "C:a\\b", false);
        t!(s: "C:\\a\\b", "\\\\?\\C:\\a\\b\\", true);
        t!(s: "\\\\?\\C:\\a\\b\\", "C:\\a\\b", true);
        t!(s: "\\\\a\\b\\c", "\\\\?\\UNC\\a\\b\\c", true);
        t!(s: "\\\\?\\UNC\\a\\b\\c", "\\\\a\\b\\c", true);
    }

    #[test]
    fn test_ends_with_path() {
        macro_rules! t(
            (s: $path:expr, $child:expr, $exp:expr) => (
                {
                    let path = Path::new($path);
                    let child = Path::new($child);
                    assert_eq!(path.ends_with_path(&child), $exp);
                }
            );
        )

        t!(s: "a\\b\\c", "c", true);
        t!(s: "a\\b\\c", "d", false);
        t!(s: "foo\\bar\\quux", "bar", false);
        t!(s: "foo\\bar\\quux", "barquux", false);
        t!(s: "a\\b\\c", "b\\c", true);
        t!(s: "a\\b\\c", "a\\b\\c", true);
        t!(s: "a\\b\\c", "foo\\a\\b\\c", false);
        t!(s: "\\a\\b\\c", "a\\b\\c", true);
        t!(s: "\\a\\b\\c", "\\a\\b\\c", false); // child must be relative
        t!(s: "\\a\\b\\c", "foo\\a\\b\\c", false);
        t!(s: "a\\b\\c", "", false);
        t!(s: "", "", true);
        t!(s: "\\a\\b\\c", "d\\e\\f", false);
        t!(s: "a\\b\\c", "a\\b", false);
        t!(s: "a\\b\\c", "b", false);
        t!(s: "C:\\a\\b", "b", true);
        t!(s: "C:\\a\\b", "C:b", false);
        t!(s: "C:\\a\\b", "C:a\\b", false);
    }

    #[test]
    fn test_path_relative_from() {
        macro_rules! t(
            (s: $path:expr, $other:expr, $exp:expr) => (
                {
                    let path = Path::new($path);
                    let other = Path::new($other);
                    let res = path.path_relative_from(&other);
                    let exp = $exp;
                    assert!(res.as_ref().and_then(|x| x.as_str()) == exp,
                            "`{}`.path_relative_from(`{}`): Expected {:?}, got {:?}",
                            path.as_str().unwrap(), other.as_str().unwrap(), exp,
                            res.as_ref().and_then(|x| x.as_str()));
                }
            )
        )

        t!(s: "a\\b\\c", "a\\b", Some("c"));
        t!(s: "a\\b\\c", "a\\b\\d", Some("..\\c"));
        t!(s: "a\\b\\c", "a\\b\\c\\d", Some(".."));
        t!(s: "a\\b\\c", "a\\b\\c", Some("."));
        t!(s: "a\\b\\c", "a\\b\\c\\d\\e", Some("..\\.."));
        t!(s: "a\\b\\c", "a\\d\\e", Some("..\\..\\b\\c"));
        t!(s: "a\\b\\c", "d\\e\\f", Some("..\\..\\..\\a\\b\\c"));
        t!(s: "a\\b\\c", "\\a\\b\\c", None);
        t!(s: "\\a\\b\\c", "a\\b\\c", Some("\\a\\b\\c"));
        t!(s: "\\a\\b\\c", "\\a\\b\\c\\d", Some(".."));
        t!(s: "\\a\\b\\c", "\\a\\b", Some("c"));
        t!(s: "\\a\\b\\c", "\\a\\b\\c\\d\\e", Some("..\\.."));
        t!(s: "\\a\\b\\c", "\\a\\d\\e", Some("..\\..\\b\\c"));
        t!(s: "\\a\\b\\c", "\\d\\e\\f", Some("..\\..\\..\\a\\b\\c"));
        t!(s: "hi\\there.txt", "hi\\there", Some("..\\there.txt"));
        t!(s: ".", "a", Some(".."));
        t!(s: ".", "a\\b", Some("..\\.."));
        t!(s: ".", ".", Some("."));
        t!(s: "a", ".", Some("a"));
        t!(s: "a\\b", ".", Some("a\\b"));
        t!(s: "..", ".", Some(".."));
        t!(s: "a\\b\\c", "a\\b\\c", Some("."));
        t!(s: "\\a\\b\\c", "\\a\\b\\c", Some("."));
        t!(s: "\\", "\\", Some("."));
        t!(s: "\\", ".", Some("\\"));
        t!(s: "..\\..\\a", "b", Some("..\\..\\..\\a"));
        t!(s: "a", "..\\..\\b", None);
        t!(s: "..\\..\\a", "..\\..\\b", Some("..\\a"));
        t!(s: "..\\..\\a", "..\\..\\a\\b", Some(".."));
        t!(s: "..\\..\\a\\b", "..\\..\\a", Some("b"));

        t!(s: "C:a\\b\\c", "C:a\\b", Some("c"));
        t!(s: "C:a\\b", "C:a\\b\\c", Some(".."));
        t!(s: "C:" ,"C:a\\b", Some("..\\.."));
        t!(s: "C:a\\b", "C:c\\d", Some("..\\..\\a\\b"));
        t!(s: "C:a\\b", "D:c\\d", Some("C:a\\b"));
        t!(s: "C:a\\b", "C:..\\c", None);
        t!(s: "C:..\\a", "C:b\\c", Some("..\\..\\..\\a"));
        t!(s: "C:\\a\\b\\c", "C:\\a\\b", Some("c"));
        t!(s: "C:\\a\\b", "C:\\a\\b\\c", Some(".."));
        t!(s: "C:\\", "C:\\a\\b", Some("..\\.."));
        t!(s: "C:\\a\\b", "C:\\c\\d", Some("..\\..\\a\\b"));
        t!(s: "C:\\a\\b", "C:a\\b", Some("C:\\a\\b"));
        t!(s: "C:a\\b", "C:\\a\\b", None);
        t!(s: "\\a\\b", "C:\\a\\b", None);
        t!(s: "\\a\\b", "C:a\\b", None);
        t!(s: "a\\b", "C:\\a\\b", None);
        t!(s: "a\\b", "C:a\\b", None);

        t!(s: "\\\\a\\b\\c", "\\\\a\\b", Some("c"));
        t!(s: "\\\\a\\b", "\\\\a\\b\\c", Some(".."));
        t!(s: "\\\\a\\b\\c\\e", "\\\\a\\b\\c\\d", Some("..\\e"));
        t!(s: "\\\\a\\c\\d", "\\\\a\\b\\d", Some("\\\\a\\c\\d"));
        t!(s: "\\\\b\\c\\d", "\\\\a\\c\\d", Some("\\\\b\\c\\d"));
        t!(s: "\\\\a\\b\\c", "\\d\\e", Some("\\\\a\\b\\c"));
        t!(s: "\\d\\e", "\\\\a\\b\\c", None);
        t!(s: "d\\e", "\\\\a\\b\\c", None);
        t!(s: "C:\\a\\b\\c", "\\\\a\\b\\c", Some("C:\\a\\b\\c"));
        t!(s: "C:\\c", "\\\\a\\b\\c", Some("C:\\c"));

        t!(s: "\\\\?\\a\\b", "\\a\\b", Some("\\\\?\\a\\b"));
        t!(s: "\\\\?\\a\\b", "a\\b", Some("\\\\?\\a\\b"));
        t!(s: "\\\\?\\a\\b", "\\b", Some("\\\\?\\a\\b"));
        t!(s: "\\\\?\\a\\b", "b", Some("\\\\?\\a\\b"));
        t!(s: "\\\\?\\a\\b", "\\\\?\\a\\b\\c", Some(".."));
        t!(s: "\\\\?\\a\\b\\c", "\\\\?\\a\\b", Some("c"));
        t!(s: "\\\\?\\a\\b", "\\\\?\\c\\d", Some("\\\\?\\a\\b"));
        t!(s: "\\\\?\\a", "\\\\?\\b", Some("\\\\?\\a"));

        t!(s: "\\\\?\\C:\\a\\b", "\\\\?\\C:\\a", Some("b"));
        t!(s: "\\\\?\\C:\\a", "\\\\?\\C:\\a\\b", Some(".."));
        t!(s: "\\\\?\\C:\\a", "\\\\?\\C:\\b", Some("..\\a"));
        t!(s: "\\\\?\\C:\\a", "\\\\?\\D:\\a", Some("\\\\?\\C:\\a"));
        t!(s: "\\\\?\\C:\\a\\b", "\\\\?\\c:\\a", Some("b"));
        t!(s: "\\\\?\\C:\\a\\b", "C:\\a", Some("b"));
        t!(s: "\\\\?\\C:\\a", "C:\\a\\b", Some(".."));
        t!(s: "C:\\a\\b", "\\\\?\\C:\\a", Some("b"));
        t!(s: "C:\\a", "\\\\?\\C:\\a\\b", Some(".."));
        t!(s: "\\\\?\\C:\\a", "D:\\a", Some("\\\\?\\C:\\a"));
        t!(s: "\\\\?\\c:\\a\\b", "C:\\a", Some("b"));
        t!(s: "\\\\?\\C:\\a\\b", "C:a\\b", Some("\\\\?\\C:\\a\\b"));
        t!(s: "\\\\?\\C:\\a\\.\\b", "C:\\a", Some("\\\\?\\C:\\a\\.\\b"));
        t!(s: "\\\\?\\C:\\a\\b/c", "C:\\a", Some("\\\\?\\C:\\a\\b/c"));
        t!(s: "\\\\?\\C:\\a\\..\\b", "C:\\a", Some("\\\\?\\C:\\a\\..\\b"));
        t!(s: "C:a\\b", "\\\\?\\C:\\a\\b", None);
        t!(s: "\\\\?\\C:\\a\\.\\b", "\\\\?\\C:\\a", Some("\\\\?\\C:\\a\\.\\b"));
        t!(s: "\\\\?\\C:\\a\\b/c", "\\\\?\\C:\\a", Some("\\\\?\\C:\\a\\b/c"));
        t!(s: "\\\\?\\C:\\a\\..\\b", "\\\\?\\C:\\a", Some("\\\\?\\C:\\a\\..\\b"));
        t!(s: "\\\\?\\C:\\a\\b\\", "\\\\?\\C:\\a", Some("b"));
        t!(s: "\\\\?\\C:\\.\\b", "\\\\?\\C:\\.", Some("b"));
        t!(s: "C:\\b", "\\\\?\\C:\\.", Some("..\\b"));
        t!(s: "\\\\?\\a\\.\\b\\c", "\\\\?\\a\\.\\b", Some("c"));
        t!(s: "\\\\?\\a\\b\\c", "\\\\?\\a\\.\\d", Some("..\\..\\b\\c"));
        t!(s: "\\\\?\\a\\..\\b", "\\\\?\\a\\..", Some("b"));
        t!(s: "\\\\?\\a\\b\\..", "\\\\?\\a\\b", Some("\\\\?\\a\\b\\.."));
        t!(s: "\\\\?\\a\\b\\c", "\\\\?\\a\\..\\b", Some("..\\..\\b\\c"));

        t!(s: "\\\\?\\UNC\\a\\b\\c", "\\\\?\\UNC\\a\\b", Some("c"));
        t!(s: "\\\\?\\UNC\\a\\b", "\\\\?\\UNC\\a\\b\\c", Some(".."));
        t!(s: "\\\\?\\UNC\\a\\b\\c", "\\\\?\\UNC\\a\\c\\d", Some("\\\\?\\UNC\\a\\b\\c"));
        t!(s: "\\\\?\\UNC\\b\\c\\d", "\\\\?\\UNC\\a\\c\\d", Some("\\\\?\\UNC\\b\\c\\d"));
        t!(s: "\\\\?\\UNC\\a\\b\\c", "\\\\?\\a\\b\\c", Some("\\\\?\\UNC\\a\\b\\c"));
        t!(s: "\\\\?\\UNC\\a\\b\\c", "\\\\?\\C:\\a\\b\\c", Some("\\\\?\\UNC\\a\\b\\c"));
        t!(s: "\\\\?\\UNC\\a\\b\\c/d", "\\\\?\\UNC\\a\\b", Some("\\\\?\\UNC\\a\\b\\c/d"));
        t!(s: "\\\\?\\UNC\\a\\b\\.", "\\\\?\\UNC\\a\\b", Some("\\\\?\\UNC\\a\\b\\."));
        t!(s: "\\\\?\\UNC\\a\\b\\..", "\\\\?\\UNC\\a\\b", Some("\\\\?\\UNC\\a\\b\\.."));
        t!(s: "\\\\?\\UNC\\a\\b\\c", "\\\\a\\b", Some("c"));
        t!(s: "\\\\?\\UNC\\a\\b", "\\\\a\\b\\c", Some(".."));
        t!(s: "\\\\?\\UNC\\a\\b\\c", "\\\\a\\c\\d", Some("\\\\?\\UNC\\a\\b\\c"));
        t!(s: "\\\\?\\UNC\\b\\c\\d", "\\\\a\\c\\d", Some("\\\\?\\UNC\\b\\c\\d"));
        t!(s: "\\\\?\\UNC\\a\\b\\.", "\\\\a\\b", Some("\\\\?\\UNC\\a\\b\\."));
        t!(s: "\\\\?\\UNC\\a\\b\\c/d", "\\\\a\\b", Some("\\\\?\\UNC\\a\\b\\c/d"));
        t!(s: "\\\\?\\UNC\\a\\b\\..", "\\\\a\\b", Some("\\\\?\\UNC\\a\\b\\.."));
        t!(s: "\\\\a\\b\\c", "\\\\?\\UNC\\a\\b", Some("c"));
        t!(s: "\\\\a\\b\\c", "\\\\?\\UNC\\a\\c\\d", Some("\\\\a\\b\\c"));
    }

    #[test]
    fn test_str_components() {
        macro_rules! t(
            (s: $path:expr, $exp:expr) => (
                {
                    let path = Path::new($path);
                    let comps = path.str_components().map(|x|x.unwrap()).to_owned_vec();
                    let exp: &[&str] = $exp;
                    assert!(comps.as_slice() == exp,
                            "str_components: Expected {:?}, found {:?}",
                            comps.as_slice(), exp);
                    let comps = path.rev_str_components().map(|x|x.unwrap()).to_owned_vec();
                    let exp = exp.rev_iter().map(|&x|x).to_owned_vec();
                    assert!(comps.as_slice() == exp,
                            "rev_str_components: Expected {:?}, found {:?}",
                            comps.as_slice(), exp);
                }
            );
            (v: [$($arg:expr),+], $exp:expr) => (
                {
                    let path = Path::new(b!($($arg),+));
                    let comps = path.str_components().map(|x|x.unwrap()).to_owned_vec();
                    let exp: &[&str] = $exp;
                    assert!(comps.as_slice() == exp,
                            "str_components: Expected {:?}, found {:?}",
                            comps.as_slice(), exp);
                    let comps = path.rev_str_components().map(|x|x.unwrap()).to_owned_vec();
                    let exp = exp.rev_iter().map(|&x|x).to_owned_vec();
                    assert!(comps.as_slice() == exp,
                            "rev_str_components: Expected {:?}, found {:?}",
                            comps.as_slice(), exp);
                }
            )
        )

        t!(v: ["a\\b\\c"], ["a", "b", "c"]);
        t!(s: "a\\b\\c", ["a", "b", "c"]);
        t!(s: "a\\b\\d", ["a", "b", "d"]);
        t!(s: "a\\b\\cd", ["a", "b", "cd"]);
        t!(s: "\\a\\b\\c", ["a", "b", "c"]);
        t!(s: "a", ["a"]);
        t!(s: "\\a", ["a"]);
        t!(s: "\\", []);
        t!(s: ".", ["."]);
        t!(s: "..", [".."]);
        t!(s: "..\\..", ["..", ".."]);
        t!(s: "..\\..\\foo", ["..", "..", "foo"]);
        t!(s: "C:foo\\bar", ["foo", "bar"]);
        t!(s: "C:foo", ["foo"]);
        t!(s: "C:", []);
        t!(s: "C:\\foo\\bar", ["foo", "bar"]);
        t!(s: "C:\\foo", ["foo"]);
        t!(s: "C:\\", []);
        t!(s: "\\\\server\\share\\foo\\bar", ["foo", "bar"]);
        t!(s: "\\\\server\\share\\foo", ["foo"]);
        t!(s: "\\\\server\\share", []);
        t!(s: "\\\\?\\foo\\bar\\baz", ["bar", "baz"]);
        t!(s: "\\\\?\\foo\\bar", ["bar"]);
        t!(s: "\\\\?\\foo", []);
        t!(s: "\\\\?\\", []);
        t!(s: "\\\\?\\a\\b", ["b"]);
        t!(s: "\\\\?\\a\\b\\", ["b"]);
        t!(s: "\\\\?\\foo\\bar\\\\baz", ["bar", "", "baz"]);
        t!(s: "\\\\?\\C:\\foo\\bar", ["foo", "bar"]);
        t!(s: "\\\\?\\C:\\foo", ["foo"]);
        t!(s: "\\\\?\\C:\\", []);
        t!(s: "\\\\?\\C:\\foo\\", ["foo"]);
        t!(s: "\\\\?\\UNC\\server\\share\\foo\\bar", ["foo", "bar"]);
        t!(s: "\\\\?\\UNC\\server\\share\\foo", ["foo"]);
        t!(s: "\\\\?\\UNC\\server\\share", []);
        t!(s: "\\\\.\\foo\\bar\\baz", ["bar", "baz"]);
        t!(s: "\\\\.\\foo\\bar", ["bar"]);
        t!(s: "\\\\.\\foo", []);
    }

    #[test]
    fn test_components_iter() {
        macro_rules! t(
            (s: $path:expr, $exp:expr) => (
                {
                    let path = Path::new($path);
                    let comps = path.components().to_owned_vec();
                    let exp: &[&[u8]] = $exp;
                    assert!(comps.as_slice() == exp, "components: Expected {:?}, found {:?}",
                            comps.as_slice(), exp);
                    let comps = path.rev_components().to_owned_vec();
                    let exp = exp.rev_iter().map(|&x|x).to_owned_vec();
                    assert!(comps.as_slice() == exp,
                            "rev_components: Expected {:?}, found {:?}",
                            comps.as_slice(), exp);
                }
            )
        )

        t!(s: "a\\b\\c", [b!("a"), b!("b"), b!("c")]);
        t!(s: ".", [b!(".")]);
        // since this is really a wrapper around str_components, those tests suffice
    }

    #[test]
    fn test_make_non_verbatim() {
        macro_rules! t(
            ($path:expr, $exp:expr) => (
                {
                    let path = Path::new($path);
                    let exp: Option<&str> = $exp;
                    let exp = exp.map(|s| Path::new(s));
                    assert!(make_non_verbatim(&path) == exp);
                }
            )
        )

        t!(r"\a\b\c", Some(r"\a\b\c"));
        t!(r"a\b\c", Some(r"a\b\c"));
        t!(r"C:\a\b\c", Some(r"C:\a\b\c"));
        t!(r"C:a\b\c", Some(r"C:a\b\c"));
        t!(r"\\server\share\foo", Some(r"\\server\share\foo"));
        t!(r"\\.\foo", None);
        t!(r"\\?\foo", None);
        t!(r"\\?\C:", None);
        t!(r"\\?\C:foo", None);
        t!(r"\\?\C:\", Some(r"C:\"));
        t!(r"\\?\C:\foo", Some(r"C:\foo"));
        t!(r"\\?\C:\foo\bar\baz", Some(r"C:\foo\bar\baz"));
        t!(r"\\?\C:\foo\.\bar\baz", None);
        t!(r"\\?\C:\foo\bar\..\baz", None);
        t!(r"\\?\C:\foo\bar\..", None);
        t!(r"\\?\UNC\server\share\foo", Some(r"\\server\share\foo"));
        t!(r"\\?\UNC\server\share", Some(r"\\server\share"));
        t!(r"\\?\UNC\server", None);
        t!(r"\\?\UNC\server\", None);
    }
}
