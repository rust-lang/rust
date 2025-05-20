//! Utilities for the `str` primitive type.
//!
//! *[See also the `str` primitive type](str).*

#![stable(feature = "rust1", since = "1.0.0")]
// Many of the usings in this module are only used in the test configuration.
// It's cleaner to just turn off the unused_imports warning than to fix them.
#![allow(unused_imports)]

use core::borrow::{Borrow, BorrowMut};
use core::iter::FusedIterator;
use core::mem::MaybeUninit;
#[stable(feature = "encode_utf16", since = "1.8.0")]
pub use core::str::EncodeUtf16;
#[stable(feature = "split_ascii_whitespace", since = "1.34.0")]
pub use core::str::SplitAsciiWhitespace;
#[stable(feature = "split_inclusive", since = "1.51.0")]
pub use core::str::SplitInclusive;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::SplitWhitespace;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::pattern;
use core::str::pattern::{DoubleEndedSearcher, Pattern, ReverseSearcher, Searcher, Utf8Pattern};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{Bytes, CharIndices, Chars, from_utf8, from_utf8_mut};
#[stable(feature = "str_escape", since = "1.34.0")]
pub use core::str::{EscapeDebug, EscapeDefault, EscapeUnicode};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{FromStr, Utf8Error};
#[allow(deprecated)]
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{Lines, LinesAny};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{MatchIndices, RMatchIndices};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{Matches, RMatches};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{ParseBoolError, from_utf8_unchecked, from_utf8_unchecked_mut};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{RSplit, Split};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{RSplitN, SplitN};
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::str::{RSplitTerminator, SplitTerminator};
#[stable(feature = "utf8_chunks", since = "1.79.0")]
pub use core::str::{Utf8Chunk, Utf8Chunks};
#[unstable(feature = "str_from_raw_parts", issue = "119206")]
pub use core::str::{from_raw_parts, from_raw_parts_mut};
use core::unicode::conversions;
use core::{mem, ptr};

use crate::borrow::ToOwned;
use crate::boxed::Box;
use crate::slice::{Concat, Join, SliceIndex};
use crate::string::String;
use crate::vec::Vec;

/// Note: `str` in `Concat<str>` is not meaningful here.
/// This type parameter of the trait only exists to enable another impl.
#[cfg(not(no_global_oom_handling))]
#[unstable(feature = "slice_concat_ext", issue = "27747")]
impl<S: Borrow<str>> Concat<str> for [S] {
    type Output = String;

    fn concat(slice: &Self) -> String {
        Join::join(slice, "")
    }
}

#[cfg(not(no_global_oom_handling))]
#[unstable(feature = "slice_concat_ext", issue = "27747")]
impl<S: Borrow<str>> Join<&str> for [S] {
    type Output = String;

    fn join(slice: &Self, sep: &str) -> String {
        unsafe { String::from_utf8_unchecked(join_generic_copy(slice, sep.as_bytes())) }
    }
}

#[cfg(not(no_global_oom_handling))]
macro_rules! specialize_for_lengths {
    ($separator:expr, $target:expr, $iter:expr; $($num:expr),*) => {{
        let mut target = $target;
        let iter = $iter;
        let sep_bytes = $separator;
        match $separator.len() {
            $(
                // loops with hardcoded sizes run much faster
                // specialize the cases with small separator lengths
                $num => {
                    for s in iter {
                        copy_slice_and_advance!(target, sep_bytes);
                        let content_bytes = s.borrow().as_ref();
                        copy_slice_and_advance!(target, content_bytes);
                    }
                },
            )*
            _ => {
                // arbitrary non-zero size fallback
                for s in iter {
                    copy_slice_and_advance!(target, sep_bytes);
                    let content_bytes = s.borrow().as_ref();
                    copy_slice_and_advance!(target, content_bytes);
                }
            }
        }
        target
    }}
}

#[cfg(not(no_global_oom_handling))]
macro_rules! copy_slice_and_advance {
    ($target:expr, $bytes:expr) => {
        let len = $bytes.len();
        let (head, tail) = { $target }.split_at_mut(len);
        head.copy_from_slice($bytes);
        $target = tail;
    };
}

// Optimized join implementation that works for both Vec<T> (T: Copy) and String's inner vec
// Currently (2018-05-13) there is a bug with type inference and specialization (see issue #36262)
// For this reason SliceConcat<T> is not specialized for T: Copy and SliceConcat<str> is the
// only user of this function. It is left in place for the time when that is fixed.
//
// the bounds for String-join are S: Borrow<str> and for Vec-join Borrow<[T]>
// [T] and str both impl AsRef<[T]> for some T
// => s.borrow().as_ref() and we always have slices
#[cfg(not(no_global_oom_handling))]
fn join_generic_copy<B, T, S>(slice: &[S], sep: &[T]) -> Vec<T>
where
    T: Copy,
    B: AsRef<[T]> + ?Sized,
    S: Borrow<B>,
{
    let sep_len = sep.len();
    let mut iter = slice.iter();

    // the first slice is the only one without a separator preceding it
    let first = match iter.next() {
        Some(first) => first,
        None => return vec![],
    };

    // compute the exact total length of the joined Vec
    // if the `len` calculation overflows, we'll panic
    // we would have run out of memory anyway and the rest of the function requires
    // the entire Vec pre-allocated for safety
    let reserved_len = sep_len
        .checked_mul(iter.len())
        .and_then(|n| {
            slice.iter().map(|s| s.borrow().as_ref().len()).try_fold(n, usize::checked_add)
        })
        .expect("attempt to join into collection with len > usize::MAX");

    // prepare an uninitialized buffer
    let mut result = Vec::with_capacity(reserved_len);
    debug_assert!(result.capacity() >= reserved_len);

    result.extend_from_slice(first.borrow().as_ref());

    unsafe {
        let pos = result.len();
        let target = result.spare_capacity_mut().get_unchecked_mut(..reserved_len - pos);

        // Convert the separator and slices to slices of MaybeUninit
        // to simplify implementation in specialize_for_lengths
        let sep_uninit = core::slice::from_raw_parts(sep.as_ptr().cast(), sep.len());
        let iter_uninit = iter.map(|it| {
            let it = it.borrow().as_ref();
            core::slice::from_raw_parts(it.as_ptr().cast(), it.len())
        });

        // copy separator and slices over without bounds checks
        // generate loops with hardcoded offsets for small separators
        // massive improvements possible (~ x2)
        let remain = specialize_for_lengths!(sep_uninit, target, iter_uninit; 0, 1, 2, 3, 4);

        // A weird borrow implementation may return different
        // slices for the length calculation and the actual copy.
        // Make sure we don't expose uninitialized bytes to the caller.
        let result_len = reserved_len - remain.len();
        result.set_len(result_len);
    }
    result
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Borrow<str> for String {
    #[inline]
    fn borrow(&self) -> &str {
        &self[..]
    }
}

#[stable(feature = "string_borrow_mut", since = "1.36.0")]
impl BorrowMut<str> for String {
    #[inline]
    fn borrow_mut(&mut self) -> &mut str {
        &mut self[..]
    }
}

#[cfg(not(no_global_oom_handling))]
#[stable(feature = "rust1", since = "1.0.0")]
impl ToOwned for str {
    type Owned = String;

    #[inline]
    fn to_owned(&self) -> String {
        unsafe { String::from_utf8_unchecked(self.as_bytes().to_owned()) }
    }

    #[inline]
    fn clone_into(&self, target: &mut String) {
        target.clear();
        target.push_str(self);
    }
}

/// Methods for string slices.
impl str {
    /// Converts a `Box<str>` into a `Box<[u8]>` without copying or allocating.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = "this is a string";
    /// let boxed_str = s.to_owned().into_boxed_str();
    /// let boxed_bytes = boxed_str.into_boxed_bytes();
    /// assert_eq!(*boxed_bytes, *s.as_bytes());
    /// ```
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "str_box_extras", since = "1.20.0")]
    #[must_use = "`self` will be dropped if the result is not used"]
    #[inline]
    pub fn into_boxed_bytes(self: Box<Self>) -> Box<[u8]> {
        self.into()
    }

    /// Replaces all matches of a pattern with another string.
    ///
    /// `replace` creates a new [`String`], and copies the data from this string slice into it.
    /// While doing so, it attempts to find matches of a pattern. If it finds any, it
    /// replaces them with the replacement string slice.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "this is old";
    ///
    /// assert_eq!("this is new", s.replace("old", "new"));
    /// assert_eq!("than an old", s.replace("is", "an"));
    /// ```
    ///
    /// When the pattern doesn't match, it returns this string slice as [`String`]:
    ///
    /// ```
    /// let s = "this is old";
    /// assert_eq!(s, s.replace("cookie monster", "little lamb"));
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[rustc_allow_incoherent_impl]
    #[must_use = "this returns the replaced string as a new allocation, \
                  without modifying the original"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn replace<P: Pattern>(&self, from: P, to: &str) -> String {
        // Fast path for replacing a single ASCII character with another.
        if let Some(from_byte) = match from.as_utf8_pattern() {
            Some(Utf8Pattern::StringPattern([from_byte])) => Some(*from_byte),
            Some(Utf8Pattern::CharPattern(c)) => c.as_ascii().map(|ascii_char| ascii_char.to_u8()),
            _ => None,
        } {
            if let [to_byte] = to.as_bytes() {
                return unsafe { replace_ascii(self.as_bytes(), from_byte, *to_byte) };
            }
        }
        // Set result capacity to self.len() when from.len() <= to.len()
        let default_capacity = match from.as_utf8_pattern() {
            Some(Utf8Pattern::StringPattern(s)) if s.len() <= to.len() => self.len(),
            Some(Utf8Pattern::CharPattern(c)) if c.len_utf8() <= to.len() => self.len(),
            _ => 0,
        };
        let mut result = String::with_capacity(default_capacity);
        let mut last_end = 0;
        for (start, part) in self.match_indices(from) {
            result.push_str(unsafe { self.get_unchecked(last_end..start) });
            result.push_str(to);
            last_end = start + part.len();
        }
        result.push_str(unsafe { self.get_unchecked(last_end..self.len()) });
        result
    }

    /// Replaces first N matches of a pattern with another string.
    ///
    /// `replacen` creates a new [`String`], and copies the data from this string slice into it.
    /// While doing so, it attempts to find matches of a pattern. If it finds any, it
    /// replaces them with the replacement string slice at most `count` times.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "foo foo 123 foo";
    /// assert_eq!("new new 123 foo", s.replacen("foo", "new", 2));
    /// assert_eq!("faa fao 123 foo", s.replacen('o', "a", 3));
    /// assert_eq!("foo foo new23 foo", s.replacen(char::is_numeric, "new", 1));
    /// ```
    ///
    /// When the pattern doesn't match, it returns this string slice as [`String`]:
    ///
    /// ```
    /// let s = "this is old";
    /// assert_eq!(s, s.replacen("cookie monster", "little lamb", 10));
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[rustc_allow_incoherent_impl]
    #[must_use = "this returns the replaced string as a new allocation, \
                  without modifying the original"]
    #[stable(feature = "str_replacen", since = "1.16.0")]
    pub fn replacen<P: Pattern>(&self, pat: P, to: &str, count: usize) -> String {
        // Hope to reduce the times of re-allocation
        let mut result = String::with_capacity(32);
        let mut last_end = 0;
        for (start, part) in self.match_indices(pat).take(count) {
            result.push_str(unsafe { self.get_unchecked(last_end..start) });
            result.push_str(to);
            last_end = start + part.len();
        }
        result.push_str(unsafe { self.get_unchecked(last_end..self.len()) });
        result
    }

    /// Returns the lowercase equivalent of this string slice, as a new [`String`].
    ///
    /// 'Lowercase' is defined according to the terms of the Unicode Derived Core Property
    /// `Lowercase`.
    ///
    /// Since some characters can expand into multiple characters when changing
    /// the case, this function returns a [`String`] instead of modifying the
    /// parameter in-place.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "HELLO";
    ///
    /// assert_eq!("hello", s.to_lowercase());
    /// ```
    ///
    /// A tricky example, with sigma:
    ///
    /// ```
    /// let sigma = "Σ";
    ///
    /// assert_eq!("σ", sigma.to_lowercase());
    ///
    /// // but at the end of a word, it's ς, not σ:
    /// let odysseus = "ὈΔΥΣΣΕΎΣ";
    ///
    /// assert_eq!("ὀδυσσεύς", odysseus.to_lowercase());
    /// ```
    ///
    /// Languages without case are not changed:
    ///
    /// ```
    /// let new_year = "农历新年";
    ///
    /// assert_eq!(new_year, new_year.to_lowercase());
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[rustc_allow_incoherent_impl]
    #[must_use = "this returns the lowercase string as a new String, \
                  without modifying the original"]
    #[stable(feature = "unicode_case_mapping", since = "1.2.0")]
    pub fn to_lowercase(&self) -> String {
        let (mut s, rest) = convert_while_ascii(self, u8::to_ascii_lowercase);

        let prefix_len = s.len();

        for (i, c) in rest.char_indices() {
            if c == 'Σ' {
                // Σ maps to σ, except at the end of a word where it maps to ς.
                // This is the only conditional (contextual) but language-independent mapping
                // in `SpecialCasing.txt`,
                // so hard-code it rather than have a generic "condition" mechanism.
                // See https://github.com/rust-lang/rust/issues/26035
                let sigma_lowercase = map_uppercase_sigma(self, prefix_len + i);
                s.push(sigma_lowercase);
            } else {
                match conversions::to_lower(c) {
                    [a, '\0', _] => s.push(a),
                    [a, b, '\0'] => {
                        s.push(a);
                        s.push(b);
                    }
                    [a, b, c] => {
                        s.push(a);
                        s.push(b);
                        s.push(c);
                    }
                }
            }
        }
        return s;

        fn map_uppercase_sigma(from: &str, i: usize) -> char {
            // See https://www.unicode.org/versions/Unicode7.0.0/ch03.pdf#G33992
            // for the definition of `Final_Sigma`.
            debug_assert!('Σ'.len_utf8() == 2);
            let is_word_final = case_ignorable_then_cased(from[..i].chars().rev())
                && !case_ignorable_then_cased(from[i + 2..].chars());
            if is_word_final { 'ς' } else { 'σ' }
        }

        fn case_ignorable_then_cased<I: Iterator<Item = char>>(iter: I) -> bool {
            use core::unicode::{Case_Ignorable, Cased};
            match iter.skip_while(|&c| Case_Ignorable(c)).next() {
                Some(c) => Cased(c),
                None => false,
            }
        }
    }

    /// Returns the uppercase equivalent of this string slice, as a new [`String`].
    ///
    /// 'Uppercase' is defined according to the terms of the Unicode Derived Core Property
    /// `Uppercase`.
    ///
    /// Since some characters can expand into multiple characters when changing
    /// the case, this function returns a [`String`] instead of modifying the
    /// parameter in-place.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "hello";
    ///
    /// assert_eq!("HELLO", s.to_uppercase());
    /// ```
    ///
    /// Scripts without case are not changed:
    ///
    /// ```
    /// let new_year = "农历新年";
    ///
    /// assert_eq!(new_year, new_year.to_uppercase());
    /// ```
    ///
    /// One character can become multiple:
    /// ```
    /// let s = "tschüß";
    ///
    /// assert_eq!("TSCHÜSS", s.to_uppercase());
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[rustc_allow_incoherent_impl]
    #[must_use = "this returns the uppercase string as a new String, \
                  without modifying the original"]
    #[stable(feature = "unicode_case_mapping", since = "1.2.0")]
    pub fn to_uppercase(&self) -> String {
        let (mut s, rest) = convert_while_ascii(self, u8::to_ascii_uppercase);

        for c in rest.chars() {
            match conversions::to_upper(c) {
                [a, '\0', _] => s.push(a),
                [a, b, '\0'] => {
                    s.push(a);
                    s.push(b);
                }
                [a, b, c] => {
                    s.push(a);
                    s.push(b);
                    s.push(c);
                }
            }
        }
        s
    }

    /// Converts a [`Box<str>`] into a [`String`] without copying or allocating.
    ///
    /// # Examples
    ///
    /// ```
    /// let string = String::from("birthday gift");
    /// let boxed_str = string.clone().into_boxed_str();
    ///
    /// assert_eq!(boxed_str.into_string(), string);
    /// ```
    #[stable(feature = "box_str", since = "1.4.0")]
    #[rustc_allow_incoherent_impl]
    #[must_use = "`self` will be dropped if the result is not used"]
    #[inline]
    pub fn into_string(self: Box<Self>) -> String {
        let slice = Box::<[u8]>::from(self);
        unsafe { String::from_utf8_unchecked(slice.into_vec()) }
    }

    /// Creates a new [`String`] by repeating a string `n` times.
    ///
    /// # Panics
    ///
    /// This function will panic if the capacity would overflow.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// assert_eq!("abc".repeat(4), String::from("abcabcabcabc"));
    /// ```
    ///
    /// A panic upon overflow:
    ///
    /// ```should_panic
    /// // this will panic at runtime
    /// let huge = "0123456789abcdef".repeat(usize::MAX);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[rustc_allow_incoherent_impl]
    #[must_use]
    #[stable(feature = "repeat_str", since = "1.16.0")]
    #[inline]
    pub fn repeat(&self, n: usize) -> String {
        unsafe { String::from_utf8_unchecked(self.as_bytes().repeat(n)) }
    }

    /// Returns a copy of this string where each character is mapped to its
    /// ASCII upper case equivalent.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To uppercase the value in-place, use [`make_ascii_uppercase`].
    ///
    /// To uppercase ASCII characters in addition to non-ASCII characters, use
    /// [`to_uppercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// let s = "Grüße, Jürgen ❤";
    ///
    /// assert_eq!("GRüßE, JüRGEN ❤", s.to_ascii_uppercase());
    /// ```
    ///
    /// [`make_ascii_uppercase`]: str::make_ascii_uppercase
    /// [`to_uppercase`]: #method.to_uppercase
    #[cfg(not(no_global_oom_handling))]
    #[rustc_allow_incoherent_impl]
    #[must_use = "to uppercase the value in-place, use `make_ascii_uppercase()`"]
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn to_ascii_uppercase(&self) -> String {
        let mut s = self.to_owned();
        s.make_ascii_uppercase();
        s
    }

    /// Returns a copy of this string where each character is mapped to its
    /// ASCII lower case equivalent.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To lowercase the value in-place, use [`make_ascii_lowercase`].
    ///
    /// To lowercase ASCII characters in addition to non-ASCII characters, use
    /// [`to_lowercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// let s = "Grüße, Jürgen ❤";
    ///
    /// assert_eq!("grüße, jürgen ❤", s.to_ascii_lowercase());
    /// ```
    ///
    /// [`make_ascii_lowercase`]: str::make_ascii_lowercase
    /// [`to_lowercase`]: #method.to_lowercase
    #[cfg(not(no_global_oom_handling))]
    #[rustc_allow_incoherent_impl]
    #[must_use = "to lowercase the value in-place, use `make_ascii_lowercase()`"]
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn to_ascii_lowercase(&self) -> String {
        let mut s = self.to_owned();
        s.make_ascii_lowercase();
        s
    }
}

/// Converts a boxed slice of bytes to a boxed string slice without checking
/// that the string contains valid UTF-8.
///
/// # Safety
///
/// * The provided bytes must contain a valid UTF-8 sequence.
///
/// # Examples
///
/// ```
/// let smile_utf8 = Box::new([226, 152, 186]);
/// let smile = unsafe { std::str::from_boxed_utf8_unchecked(smile_utf8) };
///
/// assert_eq!("☺", &*smile);
/// ```
#[stable(feature = "str_box_extras", since = "1.20.0")]
#[must_use]
#[inline]
pub unsafe fn from_boxed_utf8_unchecked(v: Box<[u8]>) -> Box<str> {
    unsafe { Box::from_raw(Box::into_raw(v) as *mut str) }
}

/// Converts leading ascii bytes in `s` by calling the `convert` function.
///
/// For better average performance, this happens in chunks of `2*size_of::<usize>()`.
///
/// Returns a tuple of the converted prefix and the remainder starting from
/// the first non-ascii character.
///
/// This function is only public so that it can be verified in a codegen test,
/// see `issue-123712-str-to-lower-autovectorization.rs`.
#[unstable(feature = "str_internals", issue = "none")]
#[doc(hidden)]
#[inline]
#[cfg(not(no_global_oom_handling))]
pub fn convert_while_ascii(s: &str, convert: fn(&u8) -> u8) -> (String, &str) {
    // Process the input in chunks of 16 bytes to enable auto-vectorization.
    // Previously the chunk size depended on the size of `usize`,
    // but on 32-bit platforms with sse or neon is also the better choice.
    // The only downside on other platforms would be a bit more loop-unrolling.
    const N: usize = 16;

    let mut slice = s.as_bytes();
    let mut out = Vec::with_capacity(slice.len());
    let mut out_slice = out.spare_capacity_mut();

    let mut ascii_prefix_len = 0_usize;
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
            out_chunk[j] = MaybeUninit::new(convert(&chunk[j]));
        }

        ascii_prefix_len += N;
        slice = unsafe { slice.get_unchecked(N..) };
        out_slice = unsafe { out_slice.get_unchecked_mut(N..) };
    }

    // handle the remainder as individual bytes
    while slice.len() > 0 {
        let byte = slice[0];
        if byte > 127 {
            break;
        }
        // SAFETY: out_slice has at least same length as input slice
        unsafe {
            *out_slice.get_unchecked_mut(0) = MaybeUninit::new(convert(&byte));
        }
        ascii_prefix_len += 1;
        slice = unsafe { slice.get_unchecked(1..) };
        out_slice = unsafe { out_slice.get_unchecked_mut(1..) };
    }

    unsafe {
        // SAFETY: ascii_prefix_len bytes have been initialized above
        out.set_len(ascii_prefix_len);

        // SAFETY: We have written only valid ascii to the output vec
        let ascii_string = String::from_utf8_unchecked(out);

        // SAFETY: we know this is a valid char boundary
        // since we only skipped over leading ascii bytes
        let rest = core::str::from_utf8_unchecked(slice);

        (ascii_string, rest)
    }
}
#[inline]
#[cfg(not(no_global_oom_handling))]
#[allow(dead_code)]
/// Faster implementation of string replacement for ASCII to ASCII cases.
/// Should produce fast vectorized code.
unsafe fn replace_ascii(utf8_bytes: &[u8], from: u8, to: u8) -> String {
    let result: Vec<u8> = utf8_bytes.iter().map(|b| if *b == from { to } else { *b }).collect();
    // SAFETY: We replaced ascii with ascii on valid utf8 strings.
    unsafe { String::from_utf8_unchecked(result) }
}
