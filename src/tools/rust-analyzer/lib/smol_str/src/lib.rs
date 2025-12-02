#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![debugger_visualizer(gdb_script_file = "gdb_smolstr_printer.py")]

extern crate alloc;

use alloc::{borrow::Cow, boxed::Box, string::String, sync::Arc};
use core::{
    borrow::Borrow,
    cmp::{self, Ordering},
    convert::Infallible,
    fmt, hash, iter, mem, ops,
    str::FromStr,
};

/// A `SmolStr` is a string type that has the following properties:
///
/// * `size_of::<SmolStr>() == 24` (therefor `== size_of::<String>()` on 64 bit platforms)
/// * `Clone` is `O(1)`
/// * Strings are stack-allocated if they are:
///     * Up to 23 bytes long
///     * Longer than 23 bytes, but substrings of `WS` (see below). Such strings consist
///       solely of consecutive newlines, followed by consecutive spaces
/// * If a string does not satisfy the aforementioned conditions, it is heap-allocated
/// * Additionally, a `SmolStr` can be explicitly created from a `&'static str` without allocation
///
/// Unlike `String`, however, `SmolStr` is immutable. The primary use case for
/// `SmolStr` is a good enough default storage for tokens of typical programming
/// languages. Strings consisting of a series of newlines, followed by a series of
/// whitespace are a typical pattern in computer programs because of indentation.
/// Note that a specialized interner might be a better solution for some use cases.
///
/// `WS`: A string of 32 newlines followed by 128 spaces.
pub struct SmolStr(Repr);

impl SmolStr {
    /// Constructs an inline variant of `SmolStr`.
    ///
    /// This never allocates.
    ///
    /// # Panics
    ///
    /// Panics if `text.len() > 23`.
    #[inline]
    pub const fn new_inline(text: &str) -> SmolStr {
        assert!(text.len() <= INLINE_CAP); // avoids bounds checks in loop

        let text = text.as_bytes();
        let mut buf = [0; INLINE_CAP];
        let mut i = 0;
        while i < text.len() {
            buf[i] = text[i];
            i += 1
        }
        SmolStr(Repr::Inline {
            // SAFETY: We know that `len` is less than or equal to the maximum value of `InlineSize`
            // as we asserted it.
            len: unsafe { InlineSize::transmute_from_u8(text.len() as u8) },
            buf,
        })
    }

    /// Constructs a `SmolStr` from a statically allocated string.
    ///
    /// This never allocates.
    #[inline(always)]
    pub const fn new_static(text: &'static str) -> SmolStr {
        // NOTE: this never uses the inline storage; if a canonical
        // representation is needed, we could check for `len() < INLINE_CAP`
        // and call `new_inline`, but this would mean an extra branch.
        SmolStr(Repr::Static(text))
    }

    /// Constructs a `SmolStr` from a `str`, heap-allocating if necessary.
    #[inline(always)]
    pub fn new(text: impl AsRef<str>) -> SmolStr {
        SmolStr(Repr::new(text.as_ref()))
    }

    /// Returns a `&str` slice of this `SmolStr`.
    #[inline(always)]
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// Returns the length of `self` in bytes.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if `self` has a length of zero bytes.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns `true` if `self` is heap-allocated.
    #[inline(always)]
    pub const fn is_heap_allocated(&self) -> bool {
        matches!(self.0, Repr::Heap(..))
    }
}

impl Clone for SmolStr {
    #[inline]
    fn clone(&self) -> Self {
        // hint for faster inline / slower heap clones
        #[cold]
        #[inline(never)]
        fn cold_clone(v: &SmolStr) -> SmolStr {
            SmolStr(v.0.clone())
        }

        if self.is_heap_allocated() {
            return cold_clone(self);
        }

        // SAFETY: We verified that the payload of `Repr` is a POD
        unsafe { core::ptr::read(self as *const SmolStr) }
    }
}

impl Default for SmolStr {
    #[inline(always)]
    fn default() -> SmolStr {
        SmolStr(Repr::Inline { len: InlineSize::_V0, buf: [0; INLINE_CAP] })
    }
}

impl ops::Deref for SmolStr {
    type Target = str;

    #[inline(always)]
    fn deref(&self) -> &str {
        self.as_str()
    }
}

// region: PartialEq implementations

impl Eq for SmolStr {}
impl PartialEq<SmolStr> for SmolStr {
    fn eq(&self, other: &SmolStr) -> bool {
        self.0.ptr_eq(&other.0) || self.as_str() == other.as_str()
    }
}

impl PartialEq<str> for SmolStr {
    #[inline(always)]
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<SmolStr> for str {
    #[inline(always)]
    fn eq(&self, other: &SmolStr) -> bool {
        other == self
    }
}

impl<'a> PartialEq<&'a str> for SmolStr {
    #[inline(always)]
    fn eq(&self, other: &&'a str) -> bool {
        self == *other
    }
}

impl PartialEq<SmolStr> for &str {
    #[inline(always)]
    fn eq(&self, other: &SmolStr) -> bool {
        *self == other
    }
}

impl PartialEq<String> for SmolStr {
    #[inline(always)]
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<SmolStr> for String {
    #[inline(always)]
    fn eq(&self, other: &SmolStr) -> bool {
        other == self
    }
}

impl<'a> PartialEq<&'a String> for SmolStr {
    #[inline(always)]
    fn eq(&self, other: &&'a String) -> bool {
        self == *other
    }
}

impl PartialEq<SmolStr> for &String {
    #[inline(always)]
    fn eq(&self, other: &SmolStr) -> bool {
        *self == other
    }
}
// endregion: PartialEq implementations

impl Ord for SmolStr {
    fn cmp(&self, other: &SmolStr) -> Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl PartialOrd for SmolStr {
    fn partial_cmp(&self, other: &SmolStr) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl hash::Hash for SmolStr {
    fn hash<H: hash::Hasher>(&self, hasher: &mut H) {
        self.as_str().hash(hasher);
    }
}

impl fmt::Debug for SmolStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl fmt::Display for SmolStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

impl iter::FromIterator<char> for SmolStr {
    fn from_iter<I: iter::IntoIterator<Item = char>>(iter: I) -> SmolStr {
        from_char_iter(iter.into_iter())
    }
}

#[inline]
fn from_char_iter(iter: impl Iterator<Item = char>) -> SmolStr {
    from_buf_and_chars([0; _], 0, iter)
}

fn from_buf_and_chars(
    mut buf: [u8; INLINE_CAP],
    buf_len: usize,
    mut iter: impl Iterator<Item = char>,
) -> SmolStr {
    let min_size = iter.size_hint().0 + buf_len;
    if min_size > INLINE_CAP {
        let heap: String =
            core::str::from_utf8(&buf[..buf_len]).unwrap().chars().chain(iter).collect();
        if heap.len() <= INLINE_CAP {
            // size hint lied
            return SmolStr::new_inline(&heap);
        }
        return SmolStr(Repr::Heap(heap.into_boxed_str().into()));
    }
    let mut len = buf_len;
    while let Some(ch) = iter.next() {
        let size = ch.len_utf8();
        if size + len > INLINE_CAP {
            let (min_remaining, _) = iter.size_hint();
            let mut heap = String::with_capacity(size + len + min_remaining);
            heap.push_str(core::str::from_utf8(&buf[..len]).unwrap());
            heap.push(ch);
            heap.extend(iter);
            return SmolStr(Repr::Heap(heap.into_boxed_str().into()));
        }
        ch.encode_utf8(&mut buf[len..]);
        len += size;
    }
    SmolStr(Repr::Inline {
        // SAFETY: We know that `len` is less than or equal to the maximum value of `InlineSize`
        // as we otherwise return early.
        len: unsafe { InlineSize::transmute_from_u8(len as u8) },
        buf,
    })
}

fn build_from_str_iter<T>(mut iter: impl Iterator<Item = T>) -> SmolStr
where
    T: AsRef<str>,
    String: iter::Extend<T>,
{
    let mut len = 0;
    let mut buf = [0u8; INLINE_CAP];
    while let Some(slice) = iter.next() {
        let slice = slice.as_ref();
        let size = slice.len();
        if size + len > INLINE_CAP {
            let mut heap = String::with_capacity(size + len);
            heap.push_str(core::str::from_utf8(&buf[..len]).unwrap());
            heap.push_str(slice);
            heap.extend(iter);
            return SmolStr(Repr::Heap(heap.into_boxed_str().into()));
        }
        buf[len..][..size].copy_from_slice(slice.as_bytes());
        len += size;
    }
    SmolStr(Repr::Inline {
        // SAFETY: We know that `len` is less than or equal to the maximum value of `InlineSize`
        // as we otherwise return early.
        len: unsafe { InlineSize::transmute_from_u8(len as u8) },
        buf,
    })
}

impl iter::FromIterator<String> for SmolStr {
    fn from_iter<I: iter::IntoIterator<Item = String>>(iter: I) -> SmolStr {
        build_from_str_iter(iter.into_iter())
    }
}

impl<'a> iter::FromIterator<&'a String> for SmolStr {
    fn from_iter<I: iter::IntoIterator<Item = &'a String>>(iter: I) -> SmolStr {
        SmolStr::from_iter(iter.into_iter().map(|x| x.as_str()))
    }
}

impl<'a> iter::FromIterator<&'a str> for SmolStr {
    fn from_iter<I: iter::IntoIterator<Item = &'a str>>(iter: I) -> SmolStr {
        build_from_str_iter(iter.into_iter())
    }
}

impl AsRef<str> for SmolStr {
    #[inline(always)]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<[u8]> for SmolStr {
    #[inline(always)]
    fn as_ref(&self) -> &[u8] {
        self.as_str().as_bytes()
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl AsRef<std::ffi::OsStr> for SmolStr {
    #[inline(always)]
    fn as_ref(&self) -> &std::ffi::OsStr {
        AsRef::<std::ffi::OsStr>::as_ref(self.as_str())
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl AsRef<std::path::Path> for SmolStr {
    #[inline(always)]
    fn as_ref(&self) -> &std::path::Path {
        AsRef::<std::path::Path>::as_ref(self.as_str())
    }
}

impl From<&str> for SmolStr {
    #[inline]
    fn from(s: &str) -> SmolStr {
        SmolStr::new(s)
    }
}

impl From<&mut str> for SmolStr {
    #[inline]
    fn from(s: &mut str) -> SmolStr {
        SmolStr::new(s)
    }
}

impl From<&String> for SmolStr {
    #[inline]
    fn from(s: &String) -> SmolStr {
        SmolStr::new(s)
    }
}

impl From<String> for SmolStr {
    #[inline(always)]
    fn from(text: String) -> Self {
        Self::new(text)
    }
}

impl From<Box<str>> for SmolStr {
    #[inline]
    fn from(s: Box<str>) -> SmolStr {
        SmolStr::new(s)
    }
}

impl From<Arc<str>> for SmolStr {
    #[inline]
    fn from(s: Arc<str>) -> SmolStr {
        let repr = Repr::new_on_stack(s.as_ref()).unwrap_or(Repr::Heap(s));
        Self(repr)
    }
}

impl<'a> From<Cow<'a, str>> for SmolStr {
    #[inline]
    fn from(s: Cow<'a, str>) -> SmolStr {
        SmolStr::new(s)
    }
}

impl From<SmolStr> for Arc<str> {
    #[inline(always)]
    fn from(text: SmolStr) -> Self {
        match text.0 {
            Repr::Heap(data) => data,
            _ => text.as_str().into(),
        }
    }
}

impl From<SmolStr> for String {
    #[inline(always)]
    fn from(text: SmolStr) -> Self {
        text.as_str().into()
    }
}

impl Borrow<str> for SmolStr {
    #[inline(always)]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl FromStr for SmolStr {
    type Err = Infallible;

    #[inline]
    fn from_str(s: &str) -> Result<SmolStr, Self::Err> {
        Ok(SmolStr::from(s))
    }
}

const INLINE_CAP: usize = InlineSize::_V23 as usize;
const N_NEWLINES: usize = 32;
const N_SPACES: usize = 128;
const WS: &str = "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n                                                                                                                                ";
const _: () = {
    assert!(WS.len() == N_NEWLINES + N_SPACES);
    assert!(WS.as_bytes()[N_NEWLINES - 1] == b'\n');
    assert!(WS.as_bytes()[N_NEWLINES] == b' ');
};

/// A [`u8`] with a bunch of niches.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
enum InlineSize {
    _V0 = 0,
    _V1,
    _V2,
    _V3,
    _V4,
    _V5,
    _V6,
    _V7,
    _V8,
    _V9,
    _V10,
    _V11,
    _V12,
    _V13,
    _V14,
    _V15,
    _V16,
    _V17,
    _V18,
    _V19,
    _V20,
    _V21,
    _V22,
    _V23,
}

impl InlineSize {
    /// SAFETY: `value` must be less than or equal to [`INLINE_CAP`]
    #[inline(always)]
    const unsafe fn transmute_from_u8(value: u8) -> Self {
        debug_assert!(value <= InlineSize::_V23 as u8);
        // SAFETY: The caller is responsible to uphold this invariant
        unsafe { mem::transmute::<u8, Self>(value) }
    }
}

#[derive(Clone, Debug)]
enum Repr {
    Inline { len: InlineSize, buf: [u8; INLINE_CAP] },
    Static(&'static str),
    Heap(Arc<str>),
}

impl Repr {
    /// This function tries to create a new Repr::Inline or Repr::Static
    /// If it isn't possible, this function returns None
    fn new_on_stack<T>(text: T) -> Option<Self>
    where
        T: AsRef<str>,
    {
        let text = text.as_ref();

        let len = text.len();
        if len <= INLINE_CAP {
            let mut buf = [0; INLINE_CAP];
            buf[..len].copy_from_slice(text.as_bytes());
            return Some(Repr::Inline {
                // SAFETY: We know that `len` is less than or equal to the maximum value of `InlineSize`
                len: unsafe { InlineSize::transmute_from_u8(len as u8) },
                buf,
            });
        }

        if len <= N_NEWLINES + N_SPACES {
            let bytes = text.as_bytes();
            let possible_newline_count = cmp::min(len, N_NEWLINES);
            let newlines =
                bytes[..possible_newline_count].iter().take_while(|&&b| b == b'\n').count();
            let possible_space_count = len - newlines;
            if possible_space_count <= N_SPACES && bytes[newlines..].iter().all(|&b| b == b' ') {
                let spaces = possible_space_count;
                let substring = &WS[N_NEWLINES - newlines..N_NEWLINES + spaces];
                return Some(Repr::Static(substring));
            }
        }
        None
    }

    fn new(text: &str) -> Self {
        Self::new_on_stack(text).unwrap_or_else(|| Repr::Heap(Arc::from(text)))
    }

    #[inline(always)]
    fn len(&self) -> usize {
        match self {
            Repr::Heap(data) => data.len(),
            Repr::Static(data) => data.len(),
            Repr::Inline { len, .. } => *len as usize,
        }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        match self {
            Repr::Heap(data) => data.is_empty(),
            Repr::Static(data) => data.is_empty(),
            &Repr::Inline { len, .. } => len as u8 == 0,
        }
    }

    #[inline]
    fn as_str(&self) -> &str {
        match self {
            Repr::Heap(data) => data,
            Repr::Static(data) => data,
            Repr::Inline { len, buf } => {
                let len = *len as usize;
                // SAFETY: len is guaranteed to be <= INLINE_CAP
                let buf = unsafe { buf.get_unchecked(..len) };
                // SAFETY: buf is guaranteed to be valid utf8 for ..len bytes
                unsafe { ::core::str::from_utf8_unchecked(buf) }
            }
        }
    }

    fn ptr_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Heap(l0), Self::Heap(r0)) => Arc::ptr_eq(l0, r0),
            (Self::Static(l0), Self::Static(r0)) => core::ptr::eq(l0, r0),
            (Self::Inline { len: l_len, buf: l_buf }, Self::Inline { len: r_len, buf: r_buf }) => {
                l_len == r_len && l_buf == r_buf
            }
            _ => false,
        }
    }
}

/// Convert value to [`SmolStr`] using [`fmt::Display`], potentially without allocating.
///
/// Almost identical to [`ToString`], but converts to `SmolStr` instead.
pub trait ToSmolStr {
    fn to_smolstr(&self) -> SmolStr;
}

/// [`str`] methods producing [`SmolStr`]s.
pub trait StrExt: private::Sealed {
    /// Returns the lowercase equivalent of this string slice as a new [`SmolStr`],
    /// potentially without allocating.
    ///
    /// See [`str::to_lowercase`].
    #[must_use = "this returns a new SmolStr without modifying the original"]
    fn to_lowercase_smolstr(&self) -> SmolStr;

    /// Returns the uppercase equivalent of this string slice as a new [`SmolStr`],
    /// potentially without allocating.
    ///
    /// See [`str::to_uppercase`].
    #[must_use = "this returns a new SmolStr without modifying the original"]
    fn to_uppercase_smolstr(&self) -> SmolStr;

    /// Returns the ASCII lowercase equivalent of this string slice as a new [`SmolStr`],
    /// potentially without allocating.
    ///
    /// See [`str::to_ascii_lowercase`].
    #[must_use = "this returns a new SmolStr without modifying the original"]
    fn to_ascii_lowercase_smolstr(&self) -> SmolStr;

    /// Returns the ASCII uppercase equivalent of this string slice as a new [`SmolStr`],
    /// potentially without allocating.
    ///
    /// See [`str::to_ascii_uppercase`].
    #[must_use = "this returns a new SmolStr without modifying the original"]
    fn to_ascii_uppercase_smolstr(&self) -> SmolStr;

    /// Replaces all matches of a &str with another &str returning a new [`SmolStr`],
    /// potentially without allocating.
    ///
    /// See [`str::replace`].
    #[must_use = "this returns a new SmolStr without modifying the original"]
    fn replace_smolstr(&self, from: &str, to: &str) -> SmolStr;

    /// Replaces first N matches of a &str with another &str returning a new [`SmolStr`],
    /// potentially without allocating.
    ///
    /// See [`str::replacen`].
    #[must_use = "this returns a new SmolStr without modifying the original"]
    fn replacen_smolstr(&self, from: &str, to: &str, count: usize) -> SmolStr;
}

impl StrExt for str {
    #[inline]
    fn to_lowercase_smolstr(&self) -> SmolStr {
        let len = self.len();
        if len <= INLINE_CAP {
            let (buf, rest) = inline_convert_while_ascii(self, u8::to_ascii_lowercase);
            from_buf_and_chars(buf, len - rest.len(), rest.chars().flat_map(|c| c.to_lowercase()))
        } else {
            self.to_lowercase().into()
        }
    }

    #[inline]
    fn to_uppercase_smolstr(&self) -> SmolStr {
        let len = self.len();
        if len <= INLINE_CAP {
            let (buf, rest) = inline_convert_while_ascii(self, u8::to_ascii_uppercase);
            from_buf_and_chars(buf, len - rest.len(), rest.chars().flat_map(|c| c.to_uppercase()))
        } else {
            self.to_uppercase().into()
        }
    }

    #[inline]
    fn to_ascii_lowercase_smolstr(&self) -> SmolStr {
        let len = self.len();
        if len <= INLINE_CAP {
            let mut buf = [0u8; INLINE_CAP];
            buf[..len].copy_from_slice(self.as_bytes());
            buf[..len].make_ascii_lowercase();
            SmolStr(Repr::Inline {
                // SAFETY: `len` is in bounds
                len: unsafe { InlineSize::transmute_from_u8(len as u8) },
                buf,
            })
        } else {
            self.to_ascii_lowercase().into()
        }
    }

    #[inline]
    fn to_ascii_uppercase_smolstr(&self) -> SmolStr {
        let len = self.len();
        if len <= INLINE_CAP {
            let mut buf = [0u8; INLINE_CAP];
            buf[..len].copy_from_slice(self.as_bytes());
            buf[..len].make_ascii_uppercase();
            SmolStr(Repr::Inline {
                // SAFETY: `len` is in bounds
                len: unsafe { InlineSize::transmute_from_u8(len as u8) },
                buf,
            })
        } else {
            self.to_ascii_uppercase().into()
        }
    }

    #[inline]
    fn replace_smolstr(&self, from: &str, to: &str) -> SmolStr {
        self.replacen_smolstr(from, to, usize::MAX)
    }

    #[inline]
    fn replacen_smolstr(&self, from: &str, to: &str, mut count: usize) -> SmolStr {
        // Fast path for replacing a single ASCII character with another inline.
        if let [from_u8] = from.as_bytes()
            && let [to_u8] = to.as_bytes()
        {
            return if self.len() <= count {
                // SAFETY: `from_u8` & `to_u8` are ascii
                unsafe { replacen_1_ascii(self, |b| if b == from_u8 { *to_u8 } else { *b }) }
            } else {
                unsafe {
                    replacen_1_ascii(self, |b| {
                        if b == from_u8 && count != 0 {
                            count -= 1;
                            *to_u8
                        } else {
                            *b
                        }
                    })
                }
            };
        }

        let mut result = SmolStrBuilder::new();
        let mut last_end = 0;
        for (start, part) in self.match_indices(from).take(count) {
            // SAFETY: `start` is guaranteed to be within the bounds of `self` as per
            // `match_indices` and last_end is always less than or equal to `start`
            result.push_str(unsafe { self.get_unchecked(last_end..start) });
            result.push_str(to);
            last_end = start + part.len();
        }
        // SAFETY: `self.len()` is guaranteed to be within the bounds of `self` and last_end is
        // always less than or equal to `self.len()`
        result.push_str(unsafe { self.get_unchecked(last_end..self.len()) });
        SmolStr::from(result)
    }
}

/// SAFETY: `map` fn must only replace ascii with ascii or return unchanged bytes.
#[inline]
unsafe fn replacen_1_ascii(src: &str, mut map: impl FnMut(&u8) -> u8) -> SmolStr {
    if src.len() <= INLINE_CAP {
        let mut buf = [0u8; INLINE_CAP];
        for (idx, b) in src.as_bytes().iter().enumerate() {
            buf[idx] = map(b);
        }
        SmolStr(Repr::Inline {
            // SAFETY: `len` is in bounds
            len: unsafe { InlineSize::transmute_from_u8(src.len() as u8) },
            buf,
        })
    } else {
        let out = src.as_bytes().iter().map(map).collect();
        // SAFETY: We replaced ascii with ascii on valid utf8 strings.
        unsafe { String::from_utf8_unchecked(out).into() }
    }
}

/// Inline version of std fn `convert_while_ascii`. `s` must have len <= 23.
#[inline]
fn inline_convert_while_ascii(s: &str, convert: fn(&u8) -> u8) -> ([u8; INLINE_CAP], &str) {
    // Process the input in chunks of 16 bytes to enable auto-vectorization.
    // Previously the chunk size depended on the size of `usize`,
    // but on 32-bit platforms with sse or neon is also the better choice.
    // The only downside on other platforms would be a bit more loop-unrolling.
    const N: usize = 16;

    debug_assert!(s.len() <= INLINE_CAP, "only for inline-able strings");

    let mut slice = s.as_bytes();
    let mut out = [0u8; INLINE_CAP];
    let mut out_slice = &mut out[..slice.len()];
    let mut is_ascii = [false; N];

    while slice.len() >= N {
        // SAFETY: checked in loop condition
        let chunk = unsafe { slice.get_unchecked(..N) };
        // SAFETY: out_slice has at least same length as input slice and gets sliced with the same offsets
        let out_chunk = unsafe { out_slice.get_unchecked_mut(..N) };

        for j in 0..N {
            is_ascii[j] = chunk[j] <= 127;
        }

        // Auto-vectorization for this check is a bit fragile, sum and comparing against the chunk
        // size gives the best result, specifically a pmovmsk instruction on x86.
        // See https://github.com/llvm/llvm-project/issues/96395 for why llvm currently does not
        // currently recognize other similar idioms.
        if is_ascii.iter().map(|x| *x as u8).sum::<u8>() as usize != N {
            break;
        }

        for j in 0..N {
            out_chunk[j] = convert(&chunk[j]);
        }

        slice = unsafe { slice.get_unchecked(N..) };
        out_slice = unsafe { out_slice.get_unchecked_mut(N..) };
    }

    // handle the remainder as individual bytes
    while !slice.is_empty() {
        let byte = slice[0];
        if byte > 127 {
            break;
        }
        // SAFETY: out_slice has at least same length as input slice
        unsafe {
            *out_slice.get_unchecked_mut(0) = convert(&byte);
        }
        slice = unsafe { slice.get_unchecked(1..) };
        out_slice = unsafe { out_slice.get_unchecked_mut(1..) };
    }

    unsafe {
        // SAFETY: we know this is a valid char boundary
        // since we only skipped over leading ascii bytes
        let rest = core::str::from_utf8_unchecked(slice);
        (out, rest)
    }
}

impl<T> ToSmolStr for T
where
    T: fmt::Display + ?Sized,
{
    fn to_smolstr(&self) -> SmolStr {
        format_smolstr!("{}", self)
    }
}

mod private {
    /// No downstream impls allowed.
    pub trait Sealed {}
    impl Sealed for str {}
}

/// Formats arguments to a [`SmolStr`], potentially without allocating.
///
/// See [`alloc::format!`] or [`format_args!`] for syntax documentation.
#[macro_export]
macro_rules! format_smolstr {
    ($($tt:tt)*) => {{
        let mut w = $crate::SmolStrBuilder::new();
        ::core::fmt::Write::write_fmt(&mut w, format_args!($($tt)*)).expect("a formatting trait implementation returned an error");
        w.finish()
    }};
}

/// A builder that can be used to efficiently build a [`SmolStr`].
///
/// This won't allocate if the final string fits into the inline buffer.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct SmolStrBuilder(SmolStrBuilderRepr);

#[derive(Clone, Debug, PartialEq, Eq)]
enum SmolStrBuilderRepr {
    Inline { len: usize, buf: [u8; INLINE_CAP] },
    Heap(String),
}

impl Default for SmolStrBuilderRepr {
    #[inline]
    fn default() -> Self {
        SmolStrBuilderRepr::Inline { buf: [0; INLINE_CAP], len: 0 }
    }
}

impl SmolStrBuilder {
    /// Creates a new empty [`SmolStrBuilder`].
    #[must_use]
    pub const fn new() -> Self {
        Self(SmolStrBuilderRepr::Inline { buf: [0; INLINE_CAP], len: 0 })
    }

    /// Builds a [`SmolStr`] from `self`.
    #[must_use]
    pub fn finish(&self) -> SmolStr {
        SmolStr(match &self.0 {
            &SmolStrBuilderRepr::Inline { len, buf } => {
                debug_assert!(len <= INLINE_CAP);
                Repr::Inline {
                    // SAFETY: We know that `value.len` is less than or equal to the maximum value of `InlineSize`
                    len: unsafe { InlineSize::transmute_from_u8(len as u8) },
                    buf,
                }
            }
            SmolStrBuilderRepr::Heap(heap) => Repr::new(heap),
        })
    }

    /// Appends the given [`char`] to the end of `self`'s buffer.
    pub fn push(&mut self, c: char) {
        match &mut self.0 {
            SmolStrBuilderRepr::Inline { len, buf } => {
                let char_len = c.len_utf8();
                let new_len = *len + char_len;
                if new_len <= INLINE_CAP {
                    c.encode_utf8(&mut buf[*len..]);
                    *len += char_len;
                } else {
                    let mut heap = String::with_capacity(new_len);
                    // copy existing inline bytes over to the heap
                    // SAFETY: inline data is guaranteed to be valid utf8 for `old_len` bytes
                    unsafe { heap.as_mut_vec().extend_from_slice(&buf[..*len]) };
                    heap.push(c);
                    self.0 = SmolStrBuilderRepr::Heap(heap);
                }
            }
            SmolStrBuilderRepr::Heap(h) => h.push(c),
        }
    }

    /// Appends a given string slice onto the end of `self`'s buffer.
    pub fn push_str(&mut self, s: &str) {
        match &mut self.0 {
            SmolStrBuilderRepr::Inline { len, buf } => {
                let old_len = *len;
                *len += s.len();

                // if the new length will fit on the stack (even if it fills it entirely)
                if *len <= INLINE_CAP {
                    buf[old_len..*len].copy_from_slice(s.as_bytes());
                    return; // skip the heap push below
                }

                let mut heap = String::with_capacity(*len);

                // copy existing inline bytes over to the heap
                // SAFETY: inline data is guaranteed to be valid utf8 for `old_len` bytes
                unsafe { heap.as_mut_vec().extend_from_slice(&buf[..old_len]) };
                heap.push_str(s);
                self.0 = SmolStrBuilderRepr::Heap(heap);
            }
            SmolStrBuilderRepr::Heap(heap) => heap.push_str(s),
        }
    }
}

impl fmt::Write for SmolStrBuilder {
    #[inline]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.push_str(s);
        Ok(())
    }
}

impl From<SmolStrBuilder> for SmolStr {
    fn from(value: SmolStrBuilder) -> Self {
        value.finish()
    }
}

#[cfg(feature = "arbitrary")]
#[cfg_attr(docsrs, doc(cfg(feature = "arbitrary")))]
impl<'a> arbitrary::Arbitrary<'a> for SmolStr {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> Result<Self, arbitrary::Error> {
        let s = <&str>::arbitrary(u)?;
        Ok(SmolStr::new(s))
    }
}

#[cfg(feature = "borsh")]
#[cfg_attr(docsrs, doc(cfg(feature = "borsh")))]
mod borsh;
#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
mod serde;

#[test]
fn from_buf_and_chars_size_hinted_heap() {
    let str = from_buf_and_chars(
        *b"abcdefghijklmnopqr00000",
        18,
        "_0x1x2x3x4x5x6x7x8x9x10x11x12x13".chars(),
    );

    assert_eq!(str, "abcdefghijklmnopqr_0x1x2x3x4x5x6x7x8x9x10x11x12x13");
}
