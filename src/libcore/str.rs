/*!
 * String manipulation
 *
 * Strings are a packed UTF-8 representation of text, stored as null
 * terminated buffers of u8 bytes.  Strings should be indexed in bytes,
 * for efficiency, but UTF-8 unsafe operations should be avoided.  For
 * some heavy-duty uses, try std::rope.
 */

use cmp::{Eq, Ord};
use libc::size_t;
use io::WriterUtil;

export
   // Creating a string
   from_bytes,
   from_byte,
   from_slice,
   from_char,
   from_chars,
   append,
   concat,
   connect,

   // Reinterpretation
   as_bytes,
   as_bytes_slice,
   as_buf,
   as_c_str,

   // Adding things to and removing things from a string
   push_str_no_overallocate,
   push_str,
   push_char,
   pop_char,
   shift_char,
   view_shift_char,
   unshift_char,
   trim_left,
   trim_right,
   trim,
   trim_left_chars,
   trim_right_chars,
   trim_chars,

   // Transforming strings
   to_bytes,
   byte_slice,
   chars,
   substr,
   slice,
   view,
   split, splitn, split_nonempty,
   split_char, splitn_char, split_char_nonempty,
   split_str, split_str_nonempty,
   lines,
   lines_any,
   words,
   to_lower,
   to_upper,
   replace,

   // Comparing strings
   eq,
   eq_slice,
   le,
   hash,

   // Iterating through strings
   all, any,
   all_between, any_between,
   map,
   each, eachi,
   each_char, each_chari,
   bytes_iter,
   chars_iter,
   split_char_iter,
   splitn_char_iter,
   words_iter,
   lines_iter,

   // Searching
   find, find_from, find_between,
   rfind, rfind_from, rfind_between,
   find_char, find_char_from, find_char_between,
   rfind_char, rfind_char_from, rfind_char_between,
   find_str, find_str_from, find_str_between,
   contains, contains_char,
   starts_with,
   ends_with,

   // String properties
   is_ascii,
   is_empty,
   is_not_empty,
   is_whitespace,
   len,
   char_len,

   // Misc
   is_utf8,
   is_utf16,
   to_utf16,
   from_utf16,
   utf16_chars,
   count_chars, count_bytes,
   utf8_char_width,
   char_range_at,
   is_char_boundary,
   char_at,
   reserve,
   reserve_at_least,
   capacity,
   escape_default,
   escape_unicode,

   unsafe,
   extensions,
   StrSlice,
   UniqueStr;

/*
Section: Creating a string
*/

/**
 * Convert a vector of bytes to a UTF-8 string
 *
 * # Failure
 *
 * Fails if invalid UTF-8
 */
pure fn from_bytes(vv: &[const u8]) -> ~str {
    assert is_utf8(vv);
    return unsafe { unsafe::from_bytes(vv) };
}

/// Copy a slice into a new unique str
pure fn from_slice(s: &str) -> ~str {
    unsafe { unsafe::slice_bytes(s, 0, len(s)) }
}

/**
 * Convert a byte to a UTF-8 string
 *
 * # Failure
 *
 * Fails if invalid UTF-8
 */
pure fn from_byte(b: u8) -> ~str {
    assert b < 128u8;
    let mut v = ~[b, 0u8];
    unsafe { ::unsafe::transmute(v) }
}

/// Appends a character at the end of a string
fn push_char(&s: ~str, ch: char) {
    unsafe {
        let code = ch as uint;
        let nb = if code < max_one_b { 1u }
        else if code < max_two_b { 2u }
        else if code < max_three_b { 3u }
        else if code < max_four_b { 4u }
        else if code < max_five_b { 5u }
        else { 6u };
        let len = len(s);
        let new_len = len + nb;
        reserve_at_least(s, new_len);
        let off = len;
        do as_buf(s) |buf, _len| {
            let buf: *mut u8 = ::unsafe::reinterpret_cast(&buf);
            if nb == 1u {
                *ptr::mut_offset(buf, off) =
                    code as u8;
            } else if nb == 2u {
                *ptr::mut_offset(buf, off) =
                    (code >> 6u & 31u | tag_two_b) as u8;
                *ptr::mut_offset(buf, off + 1u) =
                    (code & 63u | tag_cont) as u8;
            } else if nb == 3u {
                *ptr::mut_offset(buf, off) =
                    (code >> 12u & 15u | tag_three_b) as u8;
                *ptr::mut_offset(buf, off + 1u) =
                    (code >> 6u & 63u | tag_cont) as u8;
                *ptr::mut_offset(buf, off + 2u) =
                    (code & 63u | tag_cont) as u8;
            } else if nb == 4u {
                *ptr::mut_offset(buf, off) =
                    (code >> 18u & 7u | tag_four_b) as u8;
                *ptr::mut_offset(buf, off + 1u) =
                    (code >> 12u & 63u | tag_cont) as u8;
                *ptr::mut_offset(buf, off + 2u) =
                    (code >> 6u & 63u | tag_cont) as u8;
                *ptr::mut_offset(buf, off + 3u) =
                    (code & 63u | tag_cont) as u8;
            } else if nb == 5u {
                *ptr::mut_offset(buf, off) =
                    (code >> 24u & 3u | tag_five_b) as u8;
                *ptr::mut_offset(buf, off + 1u) =
                    (code >> 18u & 63u | tag_cont) as u8;
                *ptr::mut_offset(buf, off + 2u) =
                    (code >> 12u & 63u | tag_cont) as u8;
                *ptr::mut_offset(buf, off + 3u) =
                    (code >> 6u & 63u | tag_cont) as u8;
                *ptr::mut_offset(buf, off + 4u) =
                    (code & 63u | tag_cont) as u8;
            } else if nb == 6u {
                *ptr::mut_offset(buf, off) =
                    (code >> 30u & 1u | tag_six_b) as u8;
                *ptr::mut_offset(buf, off + 1u) =
                    (code >> 24u & 63u | tag_cont) as u8;
                *ptr::mut_offset(buf, off + 2u) =
                    (code >> 18u & 63u | tag_cont) as u8;
                *ptr::mut_offset(buf, off + 3u) =
                    (code >> 12u & 63u | tag_cont) as u8;
                *ptr::mut_offset(buf, off + 4u) =
                    (code >> 6u & 63u | tag_cont) as u8;
                *ptr::mut_offset(buf, off + 5u) =
                    (code & 63u | tag_cont) as u8;
            }
        }

        unsafe::set_len(s, new_len);
    }
}

/// Convert a char to a string
pure fn from_char(ch: char) -> ~str {
    let mut buf = ~"";
    unchecked { push_char(buf, ch); }
    move buf
}

/// Convert a vector of chars to a string
pure fn from_chars(chs: &[char]) -> ~str {
    let mut buf = ~"";
    unchecked {
        reserve(buf, chs.len());
        for vec::each(chs) |ch| { push_char(buf, ch); }
    }
    move buf
}

/// Appends a string slice to the back of a string, without overallocating
#[inline(always)]
fn push_str_no_overallocate(&lhs: ~str, rhs: &str) {
    unsafe {
        let llen = lhs.len();
        let rlen = rhs.len();
        reserve(lhs, llen + rlen);
        do as_buf(lhs) |lbuf, _llen| {
            do as_buf(rhs) |rbuf, _rlen| {
                let dst = ptr::offset(lbuf, llen);
                ptr::memcpy(dst, rbuf, rlen);
            }
        }
        unsafe::set_len(lhs, llen + rlen);
    }
}
/// Appends a string slice to the back of a string
#[inline(always)]
fn push_str(&lhs: ~str, rhs: &str) {
    unsafe {
        let llen = lhs.len();
        let rlen = rhs.len();
        reserve_at_least(lhs, llen + rlen);
        do as_buf(lhs) |lbuf, _llen| {
            do as_buf(rhs) |rbuf, _rlen| {
                let dst = ptr::offset(lbuf, llen);
                ptr::memcpy(dst, rbuf, rlen);
            }
        }
        unsafe::set_len(lhs, llen + rlen);
    }
}

/// Concatenate two strings together
#[inline(always)]
pure fn append(+lhs: ~str, rhs: &str) -> ~str {
    let mut v <- lhs;
    unchecked {
        push_str_no_overallocate(v, rhs);
    }
    move v
}


/// Concatenate a vector of strings
pure fn concat(v: &[~str]) -> ~str {
    let mut s: ~str = ~"";
    for vec::each(v) |ss| { unchecked { push_str(s, ss) }; }
    move s
}

/// Concatenate a vector of strings, placing a given separator between each
pure fn connect(v: &[~str], sep: &str) -> ~str {
    let mut s = ~"", first = true;
    for vec::each(v) |ss| {
        if first { first = false; } else { unchecked { push_str(s, sep); } }
        unchecked { push_str(s, ss) };
    }
    move s
}

/*
Section: Adding to and removing from a string
*/

/**
 * Remove the final character from a string and return it
 *
 * # Failure
 *
 * If the string does not contain any characters
 */
fn pop_char(&s: ~str) -> char {
    let end = len(s);
    assert end > 0u;
    let {ch, prev} = char_range_at_reverse(s, end);
    unsafe { unsafe::set_len(s, prev); }
    return ch;
}

/**
 * Remove the first character from a string and return it
 *
 * # Failure
 *
 * If the string does not contain any characters
 */
fn shift_char(&s: ~str) -> char {
    let {ch, next} = char_range_at(s, 0u);
    s = unsafe { unsafe::slice_bytes(s, next, len(s)) };
    return ch;
}

/**
 * Removes the first character from a string slice and returns it. This does
 * not allocate a new string; instead, it mutates a slice to point one
 * character beyond the character that was shifted.
 *
 * # Failure
 *
 * If the string does not contain any characters
 */
#[inline]
fn view_shift_char(s: &a/str) -> (char, &a/str) {
    let {ch, next} = char_range_at(s, 0u);
    let next_s = unsafe { unsafe::view_bytes(s, next, len(s)) };
    return (ch, next_s);
}

/// Prepend a char to a string
fn unshift_char(&s: ~str, ch: char) { s = from_char(ch) + s; }

/**
 * Returns a string with leading `chars_to_trim` removed.
 *
 * # Arguments
 *
 * * s - A string
 * * chars_to_trim - A vector of chars
 *
 */
pure fn trim_left_chars(s: &str, chars_to_trim: &[char]) -> ~str {
    if chars_to_trim.is_empty() { return from_slice(s); }

    match find(s, |c| !chars_to_trim.contains(c)) {
      None => ~"",
      Some(first) => unsafe { unsafe::slice_bytes(s, first, s.len()) }
    }
}

/**
 * Returns a string with trailing `chars_to_trim` removed.
 *
 * # Arguments
 *
 * * s - A string
 * * chars_to_trim - A vector of chars
 *
 */
pure fn trim_right_chars(s: &str, chars_to_trim: &[char]) -> ~str {
    if chars_to_trim.is_empty() { return str::from_slice(s); }

    match rfind(s, |c| !chars_to_trim.contains(c)) {
      None => ~"",
      Some(last) => {
        let {next, _} = char_range_at(s, last);
        unsafe { unsafe::slice_bytes(s, 0u, next) }
      }
    }
}

/**
 * Returns a string with leading and trailing `chars_to_trim` removed.
 *
 * # Arguments
 *
 * * s - A string
 * * chars_to_trim - A vector of chars
 *
 */
pure fn trim_chars(s: &str, chars_to_trim: &[char]) -> ~str {
    trim_left_chars(trim_right_chars(s, chars_to_trim), chars_to_trim)
}

/// Returns a string with leading whitespace removed
pure fn trim_left(s: &str) -> ~str {
    match find(s, |c| !char::is_whitespace(c)) {
      None => ~"",
      Some(first) => unsafe { unsafe::slice_bytes(s, first, len(s)) }
    }
}

/// Returns a string with trailing whitespace removed
pure fn trim_right(s: &str) -> ~str {
    match rfind(s, |c| !char::is_whitespace(c)) {
      None => ~"",
      Some(last) => {
        let {next, _} = char_range_at(s, last);
        unsafe { unsafe::slice_bytes(s, 0u, next) }
      }
    }
}

/// Returns a string with leading and trailing whitespace removed
pure fn trim(s: &str) -> ~str { trim_left(trim_right(s)) }

/*
Section: Transforming strings
*/

/**
 * Converts a string to a vector of bytes
 *
 * The result vector is not null-terminated.
 */
pure fn to_bytes(s: &str) -> ~[u8] unsafe {
    let mut s_copy = from_slice(s);
    let mut v: ~[u8] = ::unsafe::transmute(s_copy);
    vec::unsafe::set_len(v, len(s));
    move v
}

/// Work with the string as a byte slice, not including trailing null.
#[inline(always)]
pure fn byte_slice<T>(s: &str, f: fn(v: &[u8]) -> T) -> T {
    do as_buf(s) |p,n| {
        unsafe { vec::unsafe::form_slice(p, n-1u, f) }
    }
}

/// Convert a string to a vector of characters
pure fn chars(s: &str) -> ~[char] {
    let mut buf = ~[], i = 0u;
    let len = len(s);
    while i < len {
        let {ch, next} = char_range_at(s, i);
        unchecked { vec::push(buf, ch); }
        i = next;
    }
    move buf
}

/**
 * Take a substring of another.
 *
 * Returns a string containing `n` characters starting at byte offset
 * `begin`.
 */
pure fn substr(s: &str, begin: uint, n: uint) -> ~str {
    slice(s, begin, begin + count_bytes(s, begin, n))
}

/**
 * Returns a slice of the given string from the byte range [`begin`..`end`)
 *
 * Fails when `begin` and `end` do not point to valid characters or
 * beyond the last character of the string
 */
pure fn slice(s: &str, begin: uint, end: uint) -> ~str {
    assert is_char_boundary(s, begin);
    assert is_char_boundary(s, end);
    unsafe { unsafe::slice_bytes(s, begin, end) }
}

/**
 * Returns a view of the given string from the byte range [`begin`..`end`)
 *
 * Fails when `begin` and `end` do not point to valid characters or beyond
 * the last character of the string
 */
pure fn view(s: &a/str, begin: uint, end: uint) -> &a/str {
    assert is_char_boundary(s, begin);
    assert is_char_boundary(s, end);
    unsafe { unsafe::view_bytes(s, begin, end) }
}

/// Splits a string into substrings at each occurrence of a given character
pure fn split_char(s: &str, sep: char) -> ~[~str] {
    split_char_inner(s, sep, len(s), true)
}

/**
 * Splits a string into substrings at each occurrence of a given
 * character up to 'count' times
 *
 * The byte must be a valid UTF-8/ASCII byte
 */
pure fn splitn_char(s: &str, sep: char, count: uint) -> ~[~str] {
    split_char_inner(s, sep, count, true)
}

/// Like `split_char`, but omits empty strings from the returned vector
pure fn split_char_nonempty(s: &str, sep: char) -> ~[~str] {
    split_char_inner(s, sep, len(s), false)
}

pure fn split_char_inner(s: &str, sep: char, count: uint, allow_empty: bool)
    -> ~[~str] {
    if sep < 128u as char {
        let b = sep as u8, l = len(s);
        let mut result = ~[], done = 0u;
        let mut i = 0u, start = 0u;
        while i < l && done < count {
            if s[i] == b {
                if allow_empty || start < i unchecked {
                    vec::push(result,
                              unsafe { unsafe::slice_bytes(s, start, i) });
                }
                start = i + 1u;
                done += 1u;
            }
            i += 1u;
        }
        if allow_empty || start < l {
            unsafe { vec::push(result, unsafe::slice_bytes(s, start, l) ) };
        }
        move result
    } else {
        splitn(s, |cur| cur == sep, count)
    }
}


/// Splits a string into substrings using a character function
pure fn split(s: &str, sepfn: fn(char) -> bool) -> ~[~str] {
    split_inner(s, sepfn, len(s), true)
}

/**
 * Splits a string into substrings using a character function, cutting at
 * most `count` times.
 */
pure fn splitn(s: &str, sepfn: fn(char) -> bool, count: uint) -> ~[~str] {
    split_inner(s, sepfn, count, true)
}

/// Like `split`, but omits empty strings from the returned vector
pure fn split_nonempty(s: &str, sepfn: fn(char) -> bool) -> ~[~str] {
    split_inner(s, sepfn, len(s), false)
}

pure fn split_inner(s: &str, sepfn: fn(cc: char) -> bool, count: uint,
               allow_empty: bool) -> ~[~str] {
    let l = len(s);
    let mut result = ~[], i = 0u, start = 0u, done = 0u;
    while i < l && done < count {
        let {ch, next} = char_range_at(s, i);
        if sepfn(ch) {
            if allow_empty || start < i unchecked {
                vec::push(result, unsafe { unsafe::slice_bytes(s, start, i)});
            }
            start = next;
            done += 1u;
        }
        i = next;
    }
    if allow_empty || start < l unchecked {
        vec::push(result, unsafe { unsafe::slice_bytes(s, start, l) });
    }
    move result
}

// See Issue #1932 for why this is a naive search
pure fn iter_matches(s: &a/str, sep: &b/str, f: fn(uint, uint)) {
    let sep_len = len(sep), l = len(s);
    assert sep_len > 0u;
    let mut i = 0u, match_start = 0u, match_i = 0u;

    while i < l {
        if s[i] == sep[match_i] {
            if match_i == 0u { match_start = i; }
            match_i += 1u;
            // Found a match
            if match_i == sep_len {
                f(match_start, i + 1u);
                match_i = 0u;
            }
            i += 1u;
        } else {
            // Failed match, backtrack
            if match_i > 0u {
                match_i = 0u;
                i = match_start + 1u;
            } else {
                i += 1u;
            }
        }
    }
}

pure fn iter_between_matches(s: &a/str, sep: &b/str, f: fn(uint, uint)) {
    let mut last_end = 0u;
    do iter_matches(s, sep) |from, to| {
        f(last_end, from);
        last_end = to;
    }
    f(last_end, len(s));
}

/**
 * Splits a string into a vector of the substrings separated by a given string
 *
 * # Example
 *
 * ~~~
 * assert ["", "XXX", "YYY", ""] == split_str(".XXX.YYY.", ".")
 * ~~~
 */
pure fn split_str(s: &a/str, sep: &b/str) -> ~[~str] {
    let mut result = ~[];
    do iter_between_matches(s, sep) |from, to| {
        unsafe { vec::push(result, unsafe::slice_bytes(s, from, to)); }
    }
    move result
}

pure fn split_str_nonempty(s: &a/str, sep: &b/str) -> ~[~str] {
    let mut result = ~[];
    do iter_between_matches(s, sep) |from, to| {
        if to > from {
            unsafe { vec::push(result, unsafe::slice_bytes(s, from, to)); }
        }
    }
    move result
}

/**
 * Splits a string into a vector of the substrings separated by LF ('\n')
 */
pure fn lines(s: &str) -> ~[~str] { split_char(s, '\n') }

/**
 * Splits a string into a vector of the substrings separated by LF ('\n')
 * and/or CR LF ("\r\n")
 */
pure fn lines_any(s: &str) -> ~[~str] {
    vec::map(lines(s), |s| {
        let l = len(s);
        let mut cp = copy s;
        if l > 0u && s[l - 1u] == '\r' as u8 {
            unsafe { unsafe::set_len(cp, l - 1u); }
        }
        move cp
    })
}

/// Splits a string into a vector of the substrings separated by whitespace
pure fn words(s: &str) -> ~[~str] {
    split_nonempty(s, |c| char::is_whitespace(c))
}

/// Convert a string to lowercase. ASCII only
pure fn to_lower(s: &str) -> ~str {
    map(s,
        |c| unchecked{(libc::tolower(c as libc::c_char)) as char}
    )
}

/// Convert a string to uppercase. ASCII only
pure fn to_upper(s: &str) -> ~str {
    map(s,
        |c| unchecked{(libc::toupper(c as libc::c_char)) as char}
    )
}

/**
 * Replace all occurrences of one string with another
 *
 * # Arguments
 *
 * * s - The string containing substrings to replace
 * * from - The string to replace
 * * to - The replacement string
 *
 * # Return value
 *
 * The original string with all occurances of `from` replaced with `to`
 */
pure fn replace(s: &str, from: &str, to: &str) -> ~str {
    let mut result = ~"", first = true;
    do iter_between_matches(s, from) |start, end| {
        if first { first = false; } else { unchecked {push_str(result, to); }}
        unsafe { push_str(result, unsafe::slice_bytes(s, start, end)); }
    }
    move result
}

/*
Section: Comparing strings
*/

/// Bytewise slice equality
#[cfg(notest)]
#[lang="str_eq"]
pure fn eq_slice(a: &str, b: &str) -> bool {
    do as_buf(a) |ap, alen| {
        do as_buf(b) |bp, blen| {
            if (alen != blen) { false }
            else {
                unsafe {
                    libc::memcmp(ap as *libc::c_void,
                                 bp as *libc::c_void,
                                 (alen - 1) as libc::size_t) == 0
                }
            }
        }
    }
}

#[cfg(test)]
pure fn eq_slice(a: &str, b: &str) -> bool {
    do as_buf(a) |ap, alen| {
        do as_buf(b) |bp, blen| {
            if (alen != blen) { false }
            else {
                unsafe {
                    libc::memcmp(ap as *libc::c_void,
                                 bp as *libc::c_void,
                                 (alen - 1) as libc::size_t) == 0
                }
            }
        }
    }
}

/// Bytewise string equality
#[cfg(notest)]
#[lang="uniq_str_eq"]
pure fn eq(a: &~str, b: &~str) -> bool {
    eq_slice(*a, *b)
}

#[cfg(test)]
pure fn eq(a: &~str, b: &~str) -> bool {
    eq_slice(*a, *b)
}

/// Bytewise slice less than
pure fn lt(a: &str, b: &str) -> bool {
    let (a_len, b_len) = (a.len(), b.len());
    let mut end = uint::min(a_len, b_len);

    let mut i = 0;
    while i < end {
        let (c_a, c_b) = (a[i], b[i]);
        if c_a < c_b { return true; }
        if c_a > c_b { return false; }
        i += 1;
    }

    return a_len < b_len;
}

/// Bytewise less than or equal
pure fn le(a: &str, b: &str) -> bool {
    !lt(b, a)
}

/// Bytewise greater than or equal
pure fn ge(a: &str, b: &str) -> bool {
    !lt(a, b)
}

/// Bytewise greater than
pure fn gt(a: &str, b: &str) -> bool {
    !le(a, b)
}

impl &str: Eq {
    #[inline(always)]
    pure fn eq(&&other: &str) -> bool {
        eq_slice(self, other)
    }
    #[inline(always)]
    pure fn ne(&&other: &str) -> bool { !self.eq(other) }
}

impl ~str: Eq {
    #[inline(always)]
    pure fn eq(&&other: ~str) -> bool {
        eq_slice(self, other)
    }
    #[inline(always)]
    pure fn ne(&&other: ~str) -> bool { !self.eq(other) }
}

impl @str: Eq {
    #[inline(always)]
    pure fn eq(&&other: @str) -> bool {
        eq_slice(self, other)
    }
    #[inline(always)]
    pure fn ne(&&other: @str) -> bool { !self.eq(other) }
}

impl ~str : Ord {
    #[inline(always)]
    pure fn lt(&&other: ~str) -> bool { lt(self, other) }
    #[inline(always)]
    pure fn le(&&other: ~str) -> bool { le(self, other) }
    #[inline(always)]
    pure fn ge(&&other: ~str) -> bool { ge(self, other) }
    #[inline(always)]
    pure fn gt(&&other: ~str) -> bool { gt(self, other) }
}

impl &str : Ord {
    #[inline(always)]
    pure fn lt(&&other: &str) -> bool { lt(self, other) }
    #[inline(always)]
    pure fn le(&&other: &str) -> bool { le(self, other) }
    #[inline(always)]
    pure fn ge(&&other: &str) -> bool { ge(self, other) }
    #[inline(always)]
    pure fn gt(&&other: &str) -> bool { gt(self, other) }
}

impl @str : Ord {
    #[inline(always)]
    pure fn lt(&&other: @str) -> bool { lt(self, other) }
    #[inline(always)]
    pure fn le(&&other: @str) -> bool { le(self, other) }
    #[inline(always)]
    pure fn ge(&&other: @str) -> bool { ge(self, other) }
    #[inline(always)]
    pure fn gt(&&other: @str) -> bool { gt(self, other) }
}

/// String hash function
pure fn hash(s: &~str) -> uint {
    hash::hash_str(*s) as uint
}

/*
Section: Iterating through strings
*/

/**
 * Return true if a predicate matches all characters or if the string
 * contains no characters
 */
pure fn all(s: &str, it: fn(char) -> bool) -> bool {
    all_between(s, 0u, len(s), it)
}

/**
 * Return true if a predicate matches any character (and false if it
 * matches none or there are no characters)
 */
pure fn any(ss: &str, pred: fn(char) -> bool) -> bool {
    !all(ss, |cc| !pred(cc))
}

/// Apply a function to each character
pure fn map(ss: &str, ff: fn(char) -> char) -> ~str {
    let mut result = ~"";
    unchecked {
        reserve(result, len(ss));
        do chars_iter(ss) |cc| {
            str::push_char(result, ff(cc));
        }
    }
    move result
}

/// Iterate over the bytes in a string
pure fn bytes_iter(ss: &str, it: fn(u8)) {
    let mut pos = 0u;
    let len = len(ss);

    while (pos < len) {
        it(ss[pos]);
        pos += 1u;
    }
}

/// Iterate over the bytes in a string
#[inline(always)]
pure fn each(s: &str, it: fn(u8) -> bool) {
    eachi(s, |_i, b| it(b) )
}

/// Iterate over the bytes in a string, with indices
#[inline(always)]
pure fn eachi(s: &str, it: fn(uint, u8) -> bool) {
    let mut i = 0u, l = len(s);
    while (i < l) {
        if !it(i, s[i]) { break; }
        i += 1u;
    }
}

/// Iterates over the chars in a string
#[inline(always)]
pure fn each_char(s: &str, it: fn(char) -> bool) {
    each_chari(s, |_i, c| it(c))
}

/// Iterates over the chars in a string, with indices
#[inline(always)]
pure fn each_chari(s: &str, it: fn(uint, char) -> bool) {
    let mut pos = 0u, ch_pos = 0u;
    let len = len(s);
    while pos < len {
        let {ch, next} = char_range_at(s, pos);
        pos = next;
        if !it(ch_pos, ch) { break; }
        ch_pos += 1u;
    }
}

/// Iterate over the characters in a string
pure fn chars_iter(s: &str, it: fn(char)) {
    let mut pos = 0u;
    let len = len(s);
    while (pos < len) {
        let {ch, next} = char_range_at(s, pos);
        pos = next;
        it(ch);
    }
}

/// Apply a function to each substring after splitting by character
pure fn split_char_iter(ss: &str, cc: char, ff: fn(&&~str)) {
   vec::iter(split_char(ss, cc), ff)
}

/**
 * Apply a function to each substring after splitting by character, up to
 * `count` times
 */
pure fn splitn_char_iter(ss: &str, sep: char, count: uint,
                         ff: fn(&&~str)) {
   vec::iter(splitn_char(ss, sep, count), ff)
}

/// Apply a function to each word
pure fn words_iter(ss: &str, ff: fn(&&~str)) {
    vec::iter(words(ss), ff)
}

/**
 * Apply a function to each line (by '\n')
 */
pure fn lines_iter(ss: &str, ff: fn(&&~str)) {
    vec::iter(lines(ss), ff)
}

/*
Section: Searching
*/

/**
 * Returns the byte index of the first matching character
 *
 * # Arguments
 *
 * * `s` - The string to search
 * * `c` - The character to search for
 *
 * # Return value
 *
 * An `option` containing the byte index of the first matching character
 * or `none` if there is no match
 */
pure fn find_char(s: &str, c: char) -> Option<uint> {
    find_char_between(s, c, 0u, len(s))
}

/**
 * Returns the byte index of the first matching character beginning
 * from a given byte offset
 *
 * # Arguments
 *
 * * `s` - The string to search
 * * `c` - The character to search for
 * * `start` - The byte index to begin searching at, inclusive
 *
 * # Return value
 *
 * An `option` containing the byte index of the first matching character
 * or `none` if there is no match
 *
 * # Failure
 *
 * `start` must be less than or equal to `len(s)`. `start` must be the
 * index of a character boundary, as defined by `is_char_boundary`.
 */
pure fn find_char_from(s: &str, c: char, start: uint) -> Option<uint> {
    find_char_between(s, c, start, len(s))
}

/**
 * Returns the byte index of the first matching character within a given range
 *
 * # Arguments
 *
 * * `s` - The string to search
 * * `c` - The character to search for
 * * `start` - The byte index to begin searching at, inclusive
 * * `end` - The byte index to end searching at, exclusive
 *
 * # Return value
 *
 * An `option` containing the byte index of the first matching character
 * or `none` if there is no match
 *
 * # Failure
 *
 * `start` must be less than or equal to `end` and `end` must be less than
 * or equal to `len(s)`. `start` must be the index of a character boundary,
 * as defined by `is_char_boundary`.
 */
pure fn find_char_between(s: &str, c: char, start: uint, end: uint)
    -> Option<uint> {
    if c < 128u as char {
        assert start <= end;
        assert end <= len(s);
        let mut i = start;
        let b = c as u8;
        while i < end {
            if s[i] == b { return Some(i); }
            i += 1u;
        }
        return None;
    } else {
        find_between(s, start, end, |x| x == c)
    }
}

/**
 * Returns the byte index of the last matching character
 *
 * # Arguments
 *
 * * `s` - The string to search
 * * `c` - The character to search for
 *
 * # Return value
 *
 * An `option` containing the byte index of the last matching character
 * or `none` if there is no match
 */
pure fn rfind_char(s: &str, c: char) -> Option<uint> {
    rfind_char_between(s, c, len(s), 0u)
}

/**
 * Returns the byte index of the last matching character beginning
 * from a given byte offset
 *
 * # Arguments
 *
 * * `s` - The string to search
 * * `c` - The character to search for
 * * `start` - The byte index to begin searching at, exclusive
 *
 * # Return value
 *
 * An `option` containing the byte index of the last matching character
 * or `none` if there is no match
 *
 * # Failure
 *
 * `start` must be less than or equal to `len(s)`. `start` must be
 * the index of a character boundary, as defined by `is_char_boundary`.
 */
pure fn rfind_char_from(s: &str, c: char, start: uint) -> Option<uint> {
    rfind_char_between(s, c, start, 0u)
}

/**
 * Returns the byte index of the last matching character within a given range
 *
 * # Arguments
 *
 * * `s` - The string to search
 * * `c` - The character to search for
 * * `start` - The byte index to begin searching at, exclusive
 * * `end` - The byte index to end searching at, inclusive
 *
 * # Return value
 *
 * An `option` containing the byte index of the last matching character
 * or `none` if there is no match
 *
 * # Failure
 *
 * `end` must be less than or equal to `start` and `start` must be less than
 * or equal to `len(s)`. `start` must be the index of a character boundary,
 * as defined by `is_char_boundary`.
 */
pure fn rfind_char_between(s: &str, c: char, start: uint, end: uint)
    -> Option<uint> {
    if c < 128u as char {
        assert start >= end;
        assert start <= len(s);
        let mut i = start;
        let b = c as u8;
        while i > end {
            i -= 1u;
            if s[i] == b { return Some(i); }
        }
        return None;
    } else {
        rfind_between(s, start, end, |x| x == c)
    }
}

/**
 * Returns the byte index of the first character that satisfies
 * the given predicate
 *
 * # Arguments
 *
 * * `s` - The string to search
 * * `f` - The predicate to satisfy
 *
 * # Return value
 *
 * An `option` containing the byte index of the first matching character
 * or `none` if there is no match
 */
pure fn find(s: &str, f: fn(char) -> bool) -> Option<uint> {
    find_between(s, 0u, len(s), f)
}

/**
 * Returns the byte index of the first character that satisfies
 * the given predicate, beginning from a given byte offset
 *
 * # Arguments
 *
 * * `s` - The string to search
 * * `start` - The byte index to begin searching at, inclusive
 * * `f` - The predicate to satisfy
 *
 * # Return value
 *
 * An `option` containing the byte index of the first matching charactor
 * or `none` if there is no match
 *
 * # Failure
 *
 * `start` must be less than or equal to `len(s)`. `start` must be the
 * index of a character boundary, as defined by `is_char_boundary`.
 */
pure fn find_from(s: &str, start: uint, f: fn(char)
    -> bool) -> Option<uint> {
    find_between(s, start, len(s), f)
}

/**
 * Returns the byte index of the first character that satisfies
 * the given predicate, within a given range
 *
 * # Arguments
 *
 * * `s` - The string to search
 * * `start` - The byte index to begin searching at, inclusive
 * * `end` - The byte index to end searching at, exclusive
 * * `f` - The predicate to satisfy
 *
 * # Return value
 *
 * An `option` containing the byte index of the first matching character
 * or `none` if there is no match
 *
 * # Failure
 *
 * `start` must be less than or equal to `end` and `end` must be less than
 * or equal to `len(s)`. `start` must be the index of a character
 * boundary, as defined by `is_char_boundary`.
 */
pure fn find_between(s: &str, start: uint, end: uint, f: fn(char) -> bool)
    -> Option<uint> {
    assert start <= end;
    assert end <= len(s);
    assert is_char_boundary(s, start);
    let mut i = start;
    while i < end {
        let {ch, next} = char_range_at(s, i);
        if f(ch) { return Some(i); }
        i = next;
    }
    return None;
}

/**
 * Returns the byte index of the last character that satisfies
 * the given predicate
 *
 * # Arguments
 *
 * * `s` - The string to search
 * * `f` - The predicate to satisfy
 *
 * # Return value
 *
 * An option containing the byte index of the last matching character
 * or `none` if there is no match
 */
pure fn rfind(s: &str, f: fn(char) -> bool) -> Option<uint> {
    rfind_between(s, len(s), 0u, f)
}

/**
 * Returns the byte index of the last character that satisfies
 * the given predicate, beginning from a given byte offset
 *
 * # Arguments
 *
 * * `s` - The string to search
 * * `start` - The byte index to begin searching at, exclusive
 * * `f` - The predicate to satisfy
 *
 * # Return value
 *
 * An `option` containing the byte index of the last matching character
 * or `none` if there is no match
 *
 * # Failure
 *
 * `start` must be less than or equal to `len(s)', `start` must be the
 * index of a character boundary, as defined by `is_char_boundary`
 */
pure fn rfind_from(s: &str, start: uint, f: fn(char) -> bool)
    -> Option<uint> {
    rfind_between(s, start, 0u, f)
}

/**
 * Returns the byte index of the last character that satisfies
 * the given predicate, within a given range
 *
 * # Arguments
 *
 * * `s` - The string to search
 * * `start` - The byte index to begin searching at, exclusive
 * * `end` - The byte index to end searching at, inclusive
 * * `f` - The predicate to satisfy
 *
 * # Return value
 *
 * An `option` containing the byte index of the last matching character
 * or `none` if there is no match
 *
 * # Failure
 *
 * `end` must be less than or equal to `start` and `start` must be less
 * than or equal to `len(s)`. `start` must be the index of a character
 * boundary, as defined by `is_char_boundary`
 */
pure fn rfind_between(s: &str, start: uint, end: uint, f: fn(char) -> bool)
    -> Option<uint> {
    assert start >= end;
    assert start <= len(s);
    assert is_char_boundary(s, start);
    let mut i = start;
    while i > end {
        let {ch, prev} = char_range_at_reverse(s, i);
        if f(ch) { return Some(prev); }
        i = prev;
    }
    return None;
}

// Utility used by various searching functions
pure fn match_at(haystack: &a/str, needle: &b/str, at: uint) -> bool {
    let mut i = at;
    for each(needle) |c| { if haystack[i] != c { return false; } i += 1u; }
    return true;
}

/**
 * Returns the byte index of the first matching substring
 *
 * # Arguments
 *
 * * `haystack` - The string to search
 * * `needle` - The string to search for
 *
 * # Return value
 *
 * An `option` containing the byte index of the first matching substring
 * or `none` if there is no match
 */
pure fn find_str(haystack: &a/str, needle: &b/str) -> Option<uint> {
    find_str_between(haystack, needle, 0u, len(haystack))
}

/**
 * Returns the byte index of the first matching substring beginning
 * from a given byte offset
 *
 * # Arguments
 *
 * * `haystack` - The string to search
 * * `needle` - The string to search for
 * * `start` - The byte index to begin searching at, inclusive
 *
 * # Return value
 *
 * An `option` containing the byte index of the last matching character
 * or `none` if there is no match
 *
 * # Failure
 *
 * `start` must be less than or equal to `len(s)`
 */
pure fn find_str_from(haystack: &a/str, needle: &b/str, start: uint)
  -> Option<uint> {
    find_str_between(haystack, needle, start, len(haystack))
}

/**
 * Returns the byte index of the first matching substring within a given range
 *
 * # Arguments
 *
 * * `haystack` - The string to search
 * * `needle` - The string to search for
 * * `start` - The byte index to begin searching at, inclusive
 * * `end` - The byte index to end searching at, exclusive
 *
 * # Return value
 *
 * An `option` containing the byte index of the first matching character
 * or `none` if there is no match
 *
 * # Failure
 *
 * `start` must be less than or equal to `end` and `end` must be less than
 * or equal to `len(s)`.
 */
pure fn find_str_between(haystack: &a/str, needle: &b/str, start: uint,
                         end:uint)
  -> Option<uint> {
    // See Issue #1932 for why this is a naive search
    assert end <= len(haystack);
    let needle_len = len(needle);
    if needle_len == 0u { return Some(start); }
    if needle_len > end { return None; }

    let mut i = start;
    let e = end - needle_len;
    while i <= e {
        if match_at(haystack, needle, i) { return Some(i); }
        i += 1u;
    }
    return None;
}

/**
 * Returns true if one string contains another
 *
 * # Arguments
 *
 * * haystack - The string to look in
 * * needle - The string to look for
 */
pure fn contains(haystack: &a/str, needle: &b/str) -> bool {
    option::is_some(find_str(haystack, needle))
}

/**
 * Returns true if a string contains a char.
 *
 * # Arguments
 *
 * * haystack - The string to look in
 * * needle - The char to look for
 */
pure fn contains_char(haystack: &str, needle: char) -> bool {
    option::is_some(find_char(haystack, needle))
}

/**
 * Returns true if one string starts with another
 *
 * # Arguments
 *
 * * haystack - The string to look in
 * * needle - The string to look for
 */
pure fn starts_with(haystack: &a/str, needle: &b/str) -> bool {
    let haystack_len = len(haystack), needle_len = len(needle);
    if needle_len == 0u { true }
    else if needle_len > haystack_len { false }
    else { match_at(haystack, needle, 0u) }
}

/**
 * Returns true if one string ends with another
 *
 * # Arguments
 *
 * * haystack - The string to look in
 * * needle - The string to look for
 */
pure fn ends_with(haystack: &a/str, needle: &b/str) -> bool {
    let haystack_len = len(haystack), needle_len = len(needle);
    if needle_len == 0u { true }
    else if needle_len > haystack_len { false }
    else { match_at(haystack, needle, haystack_len - needle_len) }
}

/*
Section: String properties
*/

/// Determines if a string contains only ASCII characters
pure fn is_ascii(s: &str) -> bool {
    let mut i: uint = len(s);
    while i > 0u { i -= 1u; if !u8::is_ascii(s[i]) { return false; } }
    return true;
}

/// Returns true if the string has length 0
pure fn is_empty(s: &str) -> bool { len(s) == 0u }

/// Returns true if the string has length greater than 0
pure fn is_not_empty(s: &str) -> bool { !is_empty(s) }

/**
 * Returns true if the string contains only whitespace
 *
 * Whitespace characters are determined by `char::is_whitespace`
 */
pure fn is_whitespace(s: &str) -> bool {
    return all(s, char::is_whitespace);
}

/**
 * Returns true if the string contains only alphanumerics
 *
 * Alphanumeric characters are determined by `char::is_alphanumeric`
 */
fn is_alphanumeric(s: &str) -> bool {
    return all(s, char::is_alphanumeric);
}

/// Returns the string length/size in bytes not counting the null terminator
pure fn len(s: &str) -> uint {
    do as_buf(s) |_p, n| { n - 1u }
}

/// Returns the number of characters that a string holds
pure fn char_len(s: &str) -> uint { count_chars(s, 0u, len(s)) }

/*
Section: Misc
*/

/// Determines if a vector of bytes contains valid UTF-8
pure fn is_utf8(v: &[const u8]) -> bool {
    let mut i = 0u;
    let total = vec::len::<u8>(v);
    while i < total {
        let mut chsize = utf8_char_width(v[i]);
        if chsize == 0u { return false; }
        if i + chsize > total { return false; }
        i += 1u;
        while chsize > 1u {
            if v[i] & 192u8 != tag_cont_u8 { return false; }
            i += 1u;
            chsize -= 1u;
        }
    }
    return true;
}

/// Determines if a vector of `u16` contains valid UTF-16
pure fn is_utf16(v: &[u16]) -> bool {
    let len = vec::len(v);
    let mut i = 0u;
    while (i < len) {
        let u = v[i];

        if  u <= 0xD7FF_u16 || u >= 0xE000_u16 {
            i += 1u;

        } else {
            if i+1u < len { return false; }
            let u2 = v[i+1u];
            if u < 0xD7FF_u16 || u > 0xDBFF_u16 { return false; }
            if u2 < 0xDC00_u16 || u2 > 0xDFFF_u16 { return false; }
            i += 2u;
        }
    }
    return true;
}

/// Converts to a vector of `u16` encoded as UTF-16
pure fn to_utf16(s: &str) -> ~[u16] {
    let mut u = ~[];
    do chars_iter(s) |cch| {
        // Arithmetic with u32 literals is easier on the eyes than chars.
        let mut ch = cch as u32;

        if (ch & 0xFFFF_u32) == ch unchecked {
            // The BMP falls through (assuming non-surrogate, as it should)
            assert ch <= 0xD7FF_u32 || ch >= 0xE000_u32;
            vec::push(u, ch as u16)
        } else unchecked {
            // Supplementary planes break into surrogates.
            assert ch >= 0x1_0000_u32 && ch <= 0x10_FFFF_u32;
            ch -= 0x1_0000_u32;
            let w1 = 0xD800_u16 | ((ch >> 10) as u16);
            let w2 = 0xDC00_u16 | ((ch as u16) & 0x3FF_u16);
            vec::push_all(u, ~[w1, w2])
        }
    }
    move u
}

pure fn utf16_chars(v: &[u16], f: fn(char)) {
    let len = vec::len(v);
    let mut i = 0u;
    while (i < len && v[i] != 0u16) {
        let mut u = v[i];

        if  u <= 0xD7FF_u16 || u >= 0xE000_u16 {
            f(u as char);
            i += 1u;

        } else {
            let u2 = v[i+1u];
            assert u >= 0xD800_u16 && u <= 0xDBFF_u16;
            assert u2 >= 0xDC00_u16 && u2 <= 0xDFFF_u16;
            let mut c = (u - 0xD800_u16) as char;
            c = c << 10;
            c |= (u2 - 0xDC00_u16) as char;
            c |= 0x1_0000_u32 as char;
            f(c);
            i += 2u;
        }
    }
}


pure fn from_utf16(v: &[u16]) -> ~str {
    let mut buf = ~"";
    unchecked {
        reserve(buf, vec::len(v));
        utf16_chars(v, |ch| push_char(buf, ch));
    }
    move buf
}


/**
 * As char_len but for a slice of a string
 *
 * # Arguments
 *
 * * s - A valid string
 * * start - The position inside `s` where to start counting in bytes
 * * end - The position where to stop counting
 *
 * # Return value
 *
 * The number of Unicode characters in `s` between the given indices.
 */
pure fn count_chars(s: &str, start: uint, end: uint) -> uint {
    assert is_char_boundary(s, start);
    assert is_char_boundary(s, end);
    let mut i = start, len = 0u;
    while i < end {
        let {next, _} = char_range_at(s, i);
        len += 1u;
        i = next;
    }
    return len;
}

/// Counts the number of bytes taken by the `n` in `s` starting from `start`.
pure fn count_bytes(s: &b/str, start: uint, n: uint) -> uint {
    assert is_char_boundary(s, start);
    let mut end = start, cnt = n;
    let l = len(s);
    while cnt > 0u {
        assert end < l;
        let {next, _} = char_range_at(s, end);
        cnt -= 1u;
        end = next;
    }
    end - start
}

/// Given a first byte, determine how many bytes are in this UTF-8 character
pure fn utf8_char_width(b: u8) -> uint {
    let byte: uint = b as uint;
    if byte < 128u { return 1u; }
    // Not a valid start byte
    if byte < 192u { return 0u; }
    if byte < 224u { return 2u; }
    if byte < 240u { return 3u; }
    if byte < 248u { return 4u; }
    if byte < 252u { return 5u; }
    return 6u;
}

/**
 * Returns false if the index points into the middle of a multi-byte
 * character sequence.
 */
pure fn is_char_boundary(s: &str, index: uint) -> bool {
    if index == len(s) { return true; }
    let b = s[index];
    return b < 128u8 || b >= 192u8;
}

/**
 * Pluck a character out of a string and return the index of the next
 * character.
 *
 * This function can be used to iterate over the unicode characters of a
 * string.
 *
 * # Example
 *
 * ~~~
 * let s = "中华Việt Nam";
 * let i = 0u;
 * while i < str::len(s) {
 *     let {ch, next} = str::char_range_at(s, i);
 *     std::io::println(fmt!("%u: %c",i,ch));
 *     i = next;
 * }
 * ~~~
 *
 * # Example output
 *
 * ~~~
 * 0: 中
 * 3: 华
 * 6: V
 * 7: i
 * 8: ệ
 * 11: t
 * 12:
 * 13: N
 * 14: a
 * 15: m
 * ~~~
 *
 * # Arguments
 *
 * * s - The string
 * * i - The byte offset of the char to extract
 *
 * # Return value
 *
 * A record {ch: char, next: uint} containing the char value and the byte
 * index of the next unicode character.
 *
 * # Failure
 *
 * If `i` is greater than or equal to the length of the string.
 * If `i` is not the index of the beginning of a valid UTF-8 character.
 */
pure fn char_range_at(s: &str, i: uint) -> {ch: char, next: uint} {
    let b0 = s[i];
    let w = utf8_char_width(b0);
    assert (w != 0u);
    if w == 1u { return {ch: b0 as char, next: i + 1u}; }
    let mut val = 0u;
    let end = i + w;
    let mut i = i + 1u;
    while i < end {
        let byte = s[i];
        assert (byte & 192u8 == tag_cont_u8);
        val <<= 6u;
        val += (byte & 63u8) as uint;
        i += 1u;
    }
    // Clunky way to get the right bits from the first byte. Uses two shifts,
    // the first to clip off the marker bits at the left of the byte, and then
    // a second (as uint) to get it to the right position.
    val += ((b0 << ((w + 1u) as u8)) as uint) << ((w - 1u) * 6u - w - 1u);
    return {ch: val as char, next: i};
}

/// Pluck a character out of a string
pure fn char_at(s: &str, i: uint) -> char { return char_range_at(s, i).ch; }

/**
 * Given a byte position and a str, return the previous char and its position
 *
 * This function can be used to iterate over a unicode string in reverse.
 */
pure fn char_range_at_reverse(ss: &str, start: uint)
    -> {ch: char, prev: uint} {

    let mut prev = start;

    // while there is a previous byte == 10......
    while prev > 0u && ss[prev - 1u] & 192u8 == tag_cont_u8 {
        prev -= 1u;
    }

    // now refer to the initial byte of previous char
    prev -= 1u;

    let ch = char_at(ss, prev);
    return {ch:ch, prev:prev};
}

/**
 * Loop through a substring, char by char
 *
 * # Safety note
 *
 * * This function does not check whether the substring is valid.
 * * This function fails if `start` or `end` do not
 *   represent valid positions inside `s`
 *
 * # Arguments
 *
 * * s - A string to traverse. It may be empty.
 * * start - The byte offset at which to start in the string.
 * * end - The end of the range to traverse
 * * it - A block to execute with each consecutive character of `s`.
 *        Return `true` to continue, `false` to stop.
 *
 * # Return value
 *
 * `true` If execution proceeded correctly, `false` if it was interrupted,
 * that is if `it` returned `false` at any point.
 */
pure fn all_between(s: &str, start: uint, end: uint,
                    it: fn(char) -> bool) -> bool {
    assert is_char_boundary(s, start);
    let mut i = start;
    while i < end {
        let {ch, next} = char_range_at(s, i);
        if !it(ch) { return false; }
        i = next;
    }
    return true;
}

/**
 * Loop through a substring, char by char
 *
 * # Safety note
 *
 * * This function does not check whether the substring is valid.
 * * This function fails if `start` or `end` do not
 *   represent valid positions inside `s`
 *
 * # Arguments
 *
 * * s - A string to traverse. It may be empty.
 * * start - The byte offset at which to start in the string.
 * * end - The end of the range to traverse
 * * it - A block to execute with each consecutive character of `s`.
 *        Return `true` to continue, `false` to stop.
 *
 * # Return value
 *
 * `true` if `it` returns `true` for any character
 */
pure fn any_between(s: &str, start: uint, end: uint,
                    it: fn(char) -> bool) -> bool {
    !all_between(s, start, end, |c| !it(c))
}

// UTF-8 tags and ranges
const tag_cont_u8: u8 = 128u8;
const tag_cont: uint = 128u;
const max_one_b: uint = 128u;
const tag_two_b: uint = 192u;
const max_two_b: uint = 2048u;
const tag_three_b: uint = 224u;
const max_three_b: uint = 65536u;
const tag_four_b: uint = 240u;
const max_four_b: uint = 2097152u;
const tag_five_b: uint = 248u;
const max_five_b: uint = 67108864u;
const tag_six_b: uint = 252u;


/**
 * Work with the byte buffer of a string.
 *
 * Allows for unsafe manipulation of strings, which is useful for foreign
 * interop.
 *
 * # Example
 *
 * ~~~
 * let i = str::as_bytes("Hello World") { |bytes| vec::len(bytes) };
 * ~~~
 */
pure fn as_bytes<T>(s: ~str, f: fn(~[u8]) -> T) -> T {
    unsafe {
        let v: *~[u8] = ::unsafe::reinterpret_cast(&ptr::addr_of(s));
        f(*v)
    }
}

/**
 * Work with the byte buffer of a string as a byte slice.
 *
 * The byte slice does not include the null terminator.
 */
pure fn as_bytes_slice(s: &a/str) -> &a/[u8] {
    unsafe {
        let (ptr, len): (*u8, uint) = ::unsafe::reinterpret_cast(&s);
        let outgoing_tuple: (*u8, uint) = (ptr, len - 1);
        return ::unsafe::reinterpret_cast(&outgoing_tuple);
    }
}

/**
 * Work with the byte buffer of a string as a null-terminated C string.
 *
 * Allows for unsafe manipulation of strings, which is useful for foreign
 * interop. This is similar to `str::as_buf`, but guarantees null-termination.
 * If the given slice is not already null-terminated, this function will
 * allocate a temporary, copy the slice, null terminate it, and pass
 * that instead.
 *
 * # Example
 *
 * ~~~
 * let s = str::as_c_str("PATH", { |path| libc::getenv(path) });
 * ~~~
 */
pure fn as_c_str<T>(s: &str, f: fn(*libc::c_char) -> T) -> T {
    do as_buf(s) |buf, len| {
        // NB: len includes the trailing null.
        assert len > 0;
        if unsafe { *(ptr::offset(buf,len-1)) != 0 } {
            as_c_str(from_slice(s), f)
        } else {
            f(buf as *libc::c_char)
        }
    }
}


/**
 * Work with the byte buffer and length of a slice.
 *
 * The given length is one byte longer than the 'official' indexable
 * length of the string. This is to permit probing the byte past the
 * indexable area for a null byte, as is the case in slices pointing
 * to full strings, or suffixes of them.
 */
#[inline(always)]
pure fn as_buf<T>(s: &str, f: fn(*u8, uint) -> T) -> T {
    unsafe {
        let v : *(*u8,uint) = ::unsafe::reinterpret_cast(&ptr::addr_of(s));
        let (buf,len) = *v;
        f(buf, len)
    }
}

/**
 * Reserves capacity for exactly `n` bytes in the given string, not including
 * the null terminator.
 *
 * Assuming single-byte characters, the resulting string will be large
 * enough to hold a string of length `n`. To account for the null terminator,
 * the underlying buffer will have the size `n` + 1.
 *
 * If the capacity for `s` is already equal to or greater than the requested
 * capacity, then no action is taken.
 *
 * # Arguments
 *
 * * s - A string
 * * n - The number of bytes to reserve space for
 */
fn reserve(&s: ~str, n: uint) {
    unsafe {
        let v: *mut ~[u8] = ::unsafe::reinterpret_cast(&ptr::addr_of(s));
        vec::reserve(*v, n + 1);
    }
}

/**
 * Reserves capacity for at least `n` bytes in the given string, not including
 * the null terminator.
 *
 * Assuming single-byte characters, the resulting string will be large
 * enough to hold a string of length `n`. To account for the null terminator,
 * the underlying buffer will have the size `n` + 1.
 *
 * This function will over-allocate in order to amortize the allocation costs
 * in scenarios where the caller may need to repeatedly reserve additional
 * space.
 *
 * If the capacity for `s` is already equal to or greater than the requested
 * capacity, then no action is taken.
 *
 * # Arguments
 *
 * * s - A string
 * * n - The number of bytes to reserve space for
 */
fn reserve_at_least(&s: ~str, n: uint) {
    reserve(s, uint::next_power_of_two(n + 1u) - 1u)
}

/**
 * Returns the number of single-byte characters the string can hold without
 * reallocating
 */
pure fn capacity(&&s: ~str) -> uint {
    do as_bytes(s) |buf| {
        let vcap = vec::capacity(buf);
        assert vcap > 0u;
        vcap - 1u
    }
}

/// Escape each char in `s` with char::escape_default.
pure fn escape_default(s: &str) -> ~str {
    let mut out: ~str = ~"";
    unchecked {
        reserve_at_least(out, str::len(s));
        chars_iter(s, |c| push_str(out, char::escape_default(c)));
    }
    move out
}

/// Escape each char in `s` with char::escape_unicode.
pure fn escape_unicode(s: &str) -> ~str {
    let mut out: ~str = ~"";
    unchecked {
        reserve_at_least(out, str::len(s));
        chars_iter(s, |c| push_str(out, char::escape_unicode(c)));
    }
    move out
}

/// Unsafe operations
mod unsafe {
   export
      from_buf,
      from_buf_len,
      from_buf_len_nocopy,
      from_c_str,
      from_c_str_len,
      from_bytes,
      slice_bytes,
      view_bytes,
      push_byte,
      pop_byte,
      shift_byte,
      set_len;

    /// Create a Rust string from a null-terminated *u8 buffer
    unsafe fn from_buf(buf: *u8) -> ~str {
        let mut curr = buf, i = 0u;
        while *curr != 0u8 {
            i += 1u;
            curr = ptr::offset(buf, i);
        }
        return from_buf_len(buf, i);
    }

    /// Create a Rust string from a *u8 buffer of the given length
    unsafe fn from_buf_len(buf: *const u8, len: uint) -> ~str {
        let mut v: ~[mut u8] = ~[mut];
        vec::reserve(v, len + 1u);
        vec::as_buf(v, |b, _len| ptr::memcpy(b, buf as *u8, len));
        vec::unsafe::set_len(v, len);
        vec::push(v, 0u8);

        assert is_utf8(v);
        return ::unsafe::transmute(v);
    }

    /// Create a Rust string from a *u8 buffer of the given length
    /// without copying
    unsafe fn from_buf_len_nocopy(buf: &a / *u8, len: uint) -> &a / str {
        let v = (*buf, len + 1);
        assert is_utf8(::unsafe::reinterpret_cast(&v));
        return ::unsafe::transmute(v);
    }

    /// Create a Rust string from a null-terminated C string
    unsafe fn from_c_str(c_str: *libc::c_char) -> ~str {
        from_buf(::unsafe::reinterpret_cast(&c_str))
    }

    /// Create a Rust string from a `*c_char` buffer of the given length
    unsafe fn from_c_str_len(c_str: *libc::c_char, len: uint) -> ~str {
        from_buf_len(::unsafe::reinterpret_cast(&c_str), len)
    }

    /// Converts a vector of bytes to a string.
    unsafe fn from_bytes(v: &[const u8]) -> ~str {
        do vec::as_const_buf(v) |buf, len| {
            from_buf_len(buf, len)
        }
    }

    /// Converts a byte to a string.
    unsafe fn from_byte(u: u8) -> ~str { unsafe::from_bytes([u]) }

    /**
     * Takes a bytewise (not UTF-8) slice from a string.
     *
     * Returns the substring from [`begin`..`end`).
     *
     * # Failure
     *
     * If begin is greater than end.
     * If end is greater than the length of the string.
     */
    unsafe fn slice_bytes(s: &str, begin: uint, end: uint) -> ~str {
        do as_buf(s) |sbuf, n| {
            assert (begin <= end);
            assert (end <= n);

            let mut v = ~[];
            vec::reserve(v, end - begin + 1u);
            unsafe {
                do vec::as_buf(v) |vbuf, _vlen| {
                    let src = ptr::offset(sbuf, begin);
                    ptr::memcpy(vbuf, src, end - begin);
                }
                vec::unsafe::set_len(v, end - begin);
                vec::push(v, 0u8);
                ::unsafe::transmute(v)
            }
        }
    }

    /**
     * Takes a bytewise (not UTF-8) view from a string.
     *
     * Returns the substring from [`begin`..`end`).
     *
     * # Failure
     *
     * If begin is greater than end.
     * If end is greater than the length of the string.
     */
    #[inline]
    unsafe fn view_bytes(s: &str, begin: uint, end: uint) -> &str {
        do as_buf(s) |sbuf, n| {
             assert (begin <= end);
             assert (end <= n);

             let tuple = (ptr::offset(sbuf, begin), end - begin + 1);
             ::unsafe::reinterpret_cast(&tuple)
        }
    }

    /// Appends a byte to a string. (Not UTF-8 safe).
    unsafe fn push_byte(&s: ~str, b: u8) {
        reserve_at_least(s, s.len() + 1);
        do as_buf(s) |buf, len| {
            let buf: *mut u8 = ::unsafe::reinterpret_cast(&buf);
            *ptr::mut_offset(buf, len) = b;
        }
        set_len(s, s.len() + 1);
    }

    /// Appends a vector of bytes to a string. (Not UTF-8 safe).
    unsafe fn push_bytes(&s: ~str, bytes: ~[u8]) {
        reserve_at_least(s, s.len() + bytes.len());
        for vec::each(bytes) |byte| { push_byte(s, byte); }
    }

    /// Removes the last byte from a string and returns it. (Not UTF-8 safe).
    unsafe fn pop_byte(&s: ~str) -> u8 {
        let len = len(s);
        assert (len > 0u);
        let b = s[len - 1u];
        unsafe { set_len(s, len - 1u) };
        return b;
    }

    /// Removes the first byte from a string and returns it. (Not UTF-8 safe).
    unsafe fn shift_byte(&s: ~str) -> u8 {
        let len = len(s);
        assert (len > 0u);
        let b = s[0];
        s = unsafe { unsafe::slice_bytes(s, 1u, len) };
        return b;
    }

    /// Sets the length of the string and adds the null terminator
    unsafe fn set_len(&v: ~str, new_len: uint) {
        let repr: *vec::unsafe::VecRepr = ::unsafe::reinterpret_cast(&v);
        (*repr).fill = new_len + 1u;
        let null = ptr::mut_offset(ptr::mut_addr_of((*repr).data), new_len);
        *null = 0u8;
    }

    #[test]
    fn test_from_buf_len() {
        unsafe {
            let a = ~[65u8, 65u8, 65u8, 65u8, 65u8, 65u8, 65u8, 0u8];
            let b = vec::unsafe::to_ptr(a);
            let c = from_buf_len(b, 3u);
            assert (c == ~"AAA");
        }
    }

}

trait UniqueStr {
    fn trim() -> self;
    fn trim_left() -> self;
    fn trim_right() -> self;
}

/// Extension methods for strings
impl ~str: UniqueStr {
    /// Returns a string with leading and trailing whitespace removed
    #[inline]
    fn trim() -> ~str { trim(self) }
    /// Returns a string with leading whitespace removed
    #[inline]
    fn trim_left() -> ~str { trim_left(self) }
    /// Returns a string with trailing whitespace removed
    #[inline]
    fn trim_right() -> ~str { trim_right(self) }
}

#[cfg(notest)]
impl ~str: Add<&str,~str> {
    #[inline(always)]
    pure fn add(rhs: &str) -> ~str {
        append(copy self, rhs)
    }
}

trait StrSlice {
    fn all(it: fn(char) -> bool) -> bool;
    fn any(it: fn(char) -> bool) -> bool;
    fn contains(needle: &a/str) -> bool;
    fn contains_char(needle: char) -> bool;
    fn each(it: fn(u8) -> bool);
    fn eachi(it: fn(uint, u8) -> bool);
    fn each_char(it: fn(char) -> bool);
    fn each_chari(it: fn(uint, char) -> bool);
    fn ends_with(needle: &str) -> bool;
    fn is_empty() -> bool;
    fn is_not_empty() -> bool;
    fn is_whitespace() -> bool;
    fn is_alphanumeric() -> bool;
    pure fn len() -> uint;
    pure fn slice(begin: uint, end: uint) -> ~str;
    fn split(sepfn: fn(char) -> bool) -> ~[~str];
    fn split_char(sep: char) -> ~[~str];
    fn split_str(sep: &a/str) -> ~[~str];
    fn starts_with(needle: &a/str) -> bool;
    fn substr(begin: uint, n: uint) -> ~str;
    pure fn to_lower() -> ~str;
    pure fn to_upper() -> ~str;
    fn escape_default() -> ~str;
    fn escape_unicode() -> ~str;
    pure fn to_unique() -> ~str;
    pure fn char_at(i: uint) -> char;
}

/// Extension methods for strings
impl &str: StrSlice {
    /**
     * Return true if a predicate matches all characters or if the string
     * contains no characters
     */
    #[inline]
    fn all(it: fn(char) -> bool) -> bool { all(self, it) }
    /**
     * Return true if a predicate matches any character (and false if it
     * matches none or there are no characters)
     */
    #[inline]
    fn any(it: fn(char) -> bool) -> bool { any(self, it) }
    /// Returns true if one string contains another
    #[inline]
    fn contains(needle: &a/str) -> bool { contains(self, needle) }
    /// Returns true if a string contains a char
    #[inline]
    fn contains_char(needle: char) -> bool { contains_char(self, needle) }
    /// Iterate over the bytes in a string
    #[inline]
    fn each(it: fn(u8) -> bool) { each(self, it) }
    /// Iterate over the bytes in a string, with indices
    #[inline]
    fn eachi(it: fn(uint, u8) -> bool) { eachi(self, it) }
    /// Iterate over the chars in a string
    #[inline]
    fn each_char(it: fn(char) -> bool) { each_char(self, it) }
    /// Iterate over the chars in a string, with indices
    #[inline]
    fn each_chari(it: fn(uint, char) -> bool) { each_chari(self, it) }
    /// Returns true if one string ends with another
    #[inline]
    fn ends_with(needle: &str) -> bool { ends_with(self, needle) }
    /// Returns true if the string has length 0
    #[inline]
    fn is_empty() -> bool { is_empty(self) }
    /// Returns true if the string has length greater than 0
    #[inline]
    fn is_not_empty() -> bool { is_not_empty(self) }
    /**
     * Returns true if the string contains only whitespace
     *
     * Whitespace characters are determined by `char::is_whitespace`
     */
    #[inline]
    fn is_whitespace() -> bool { is_whitespace(self) }
    /**
     * Returns true if the string contains only alphanumerics
     *
     * Alphanumeric characters are determined by `char::is_alphanumeric`
     */
    #[inline]
    fn is_alphanumeric() -> bool { is_alphanumeric(self) }
    #[inline]
    /// Returns the size in bytes not counting the null terminator
    pure fn len() -> uint { len(self) }
    /**
     * Returns a slice of the given string from the byte range
     * [`begin`..`end`)
     *
     * Fails when `begin` and `end` do not point to valid characters or
     * beyond the last character of the string
     */
    #[inline]
    pure fn slice(begin: uint, end: uint) -> ~str { slice(self, begin, end) }
    /// Splits a string into substrings using a character function
    #[inline]
    fn split(sepfn: fn(char) -> bool) -> ~[~str] { split(self, sepfn) }
    /**
     * Splits a string into substrings at each occurrence of a given character
     */
    #[inline]
    fn split_char(sep: char) -> ~[~str] { split_char(self, sep) }
    /**
     * Splits a string into a vector of the substrings separated by a given
     * string
     */
    #[inline]
    fn split_str(sep: &a/str) -> ~[~str] { split_str(self, sep) }
    /// Returns true if one string starts with another
    #[inline]
    fn starts_with(needle: &a/str) -> bool { starts_with(self, needle) }
    /**
     * Take a substring of another.
     *
     * Returns a string containing `n` characters starting at byte offset
     * `begin`.
     */
    #[inline]
    fn substr(begin: uint, n: uint) -> ~str { substr(self, begin, n) }
    /// Convert a string to lowercase
    #[inline]
    pure fn to_lower() -> ~str { to_lower(self) }
    /// Convert a string to uppercase
    #[inline]
    pure fn to_upper() -> ~str { to_upper(self) }
    /// Escape each char in `s` with char::escape_default.
    #[inline]
    fn escape_default() -> ~str { escape_default(self) }
    /// Escape each char in `s` with char::escape_unicode.
    #[inline]
    fn escape_unicode() -> ~str { escape_unicode(self) }

    #[inline]
    pure fn to_unique() -> ~str { self.slice(0, self.len()) }

    #[inline]
    pure fn char_at(i: uint) -> char { char_at(self, i) }
}

#[cfg(test)]
mod tests {

    import libc::c_char;

    #[test]
    fn test_eq() {
        assert (eq(&~"", &~""));
        assert (eq(&~"foo", &~"foo"));
        assert (!eq(&~"foo", &~"bar"));
    }

    #[test]
    fn test_eq_slice() {
        assert (eq_slice(view("foobar", 0, 3), "foo"));
        assert (eq_slice(view("barfoo", 3, 6), "foo"));
        assert (!eq_slice("foo1", "foo2"));
    }

    #[test]
    fn test_le() {
        assert (le(&"", &""));
        assert (le(&"", &"foo"));
        assert (le(&"foo", &"foo"));
        assert (!eq(&~"foo", &~"bar"));
    }

    #[test]
    fn test_len() {
        assert (len(~"") == 0u);
        assert (len(~"hello world") == 11u);
        assert (len(~"\x63") == 1u);
        assert (len(~"\xa2") == 2u);
        assert (len(~"\u03c0") == 2u);
        assert (len(~"\u2620") == 3u);
        assert (len(~"\U0001d11e") == 4u);

        assert (char_len(~"") == 0u);
        assert (char_len(~"hello world") == 11u);
        assert (char_len(~"\x63") == 1u);
        assert (char_len(~"\xa2") == 1u);
        assert (char_len(~"\u03c0") == 1u);
        assert (char_len(~"\u2620") == 1u);
        assert (char_len(~"\U0001d11e") == 1u);
        assert (char_len(~"ประเทศไทย中华Việt Nam") == 19u);
    }

    #[test]
    fn test_rfind_char() {
        assert rfind_char(~"hello", 'l') == Some(3u);
        assert rfind_char(~"hello", 'o') == Some(4u);
        assert rfind_char(~"hello", 'h') == Some(0u);
        assert rfind_char(~"hello", 'z').is_none();
        assert rfind_char(~"ประเทศไทย中华Việt Nam", '华') == Some(30u);
    }

    #[test]
    fn test_pop_char() {
        let mut data = ~"ประเทศไทย中华";
        let cc = pop_char(data);
        assert ~"ประเทศไทย中" == data;
        assert '华' == cc;
    }

    #[test]
    fn test_pop_char_2() {
        let mut data2 = ~"华";
        let cc2 = pop_char(data2);
        assert ~"" == data2;
        assert '华' == cc2;
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_pop_char_fail() {
        let mut data = ~"";
        let _cc3 = pop_char(data);
    }

    #[test]
    fn test_split_char() {
        fn t(s: ~str, c: char, u: ~[~str]) {
            log(debug, ~"split_byte: " + s);
            let v = split_char(s, c);
            debug!("split_byte to: %?", v);
            assert vec::all2(v, u, |a,b| a == b);
        }
        t(~"abc.hello.there", '.', ~[~"abc", ~"hello", ~"there"]);
        t(~".hello.there", '.', ~[~"", ~"hello", ~"there"]);
        t(~"...hello.there.", '.', ~[~"", ~"", ~"", ~"hello", ~"there", ~""]);

        assert ~[~"", ~"", ~"", ~"hello", ~"there", ~""]
            == split_char(~"...hello.there.", '.');

        assert ~[~""] == split_char(~"", 'z');
        assert ~[~"",~""] == split_char(~"z", 'z');
        assert ~[~"ok"] == split_char(~"ok", 'z');
    }

    #[test]
    fn test_split_char_2() {
        let data = ~"ประเทศไทย中华Việt Nam";
        assert ~[~"ประเทศไทย中华", ~"iệt Nam"]
            == split_char(data, 'V');
        assert ~[~"ประเ", ~"ศไ", ~"ย中华Việt Nam"]
            == split_char(data, 'ท');
    }

    #[test]
    fn test_splitn_char() {
        fn t(s: ~str, c: char, n: uint, u: ~[~str]) {
            log(debug, ~"splitn_byte: " + s);
            let v = splitn_char(s, c, n);
            debug!("split_byte to: %?", v);
            debug!("comparing vs. %?", u);
            assert vec::all2(v, u, |a,b| a == b);
        }
        t(~"abc.hello.there", '.', 0u, ~[~"abc.hello.there"]);
        t(~"abc.hello.there", '.', 1u, ~[~"abc", ~"hello.there"]);
        t(~"abc.hello.there", '.', 2u, ~[~"abc", ~"hello", ~"there"]);
        t(~"abc.hello.there", '.', 3u, ~[~"abc", ~"hello", ~"there"]);
        t(~".hello.there", '.', 0u, ~[~".hello.there"]);
        t(~".hello.there", '.', 1u, ~[~"", ~"hello.there"]);
        t(~"...hello.there.", '.', 3u, ~[~"", ~"", ~"", ~"hello.there."]);
        t(~"...hello.there.", '.', 5u,
          ~[~"", ~"", ~"", ~"hello", ~"there", ~""]);

        assert ~[~""] == splitn_char(~"", 'z', 5u);
        assert ~[~"",~""] == splitn_char(~"z", 'z', 5u);
        assert ~[~"ok"] == splitn_char(~"ok", 'z', 5u);
        assert ~[~"z"] == splitn_char(~"z", 'z', 0u);
        assert ~[~"w.x.y"] == splitn_char(~"w.x.y", '.', 0u);
        assert ~[~"w",~"x.y"] == splitn_char(~"w.x.y", '.', 1u);
    }

    #[test]
    fn test_splitn_char_2 () {
        let data = ~"ประเทศไทย中华Việt Nam";
        assert ~[~"ประเทศไทย中", ~"Việt Nam"]
            == splitn_char(data, '华', 1u);

        assert ~[~"", ~"", ~"XXX", ~"YYYzWWWz"]
            == splitn_char(~"zzXXXzYYYzWWWz", 'z', 3u);
        assert ~[~"",~""] == splitn_char(~"z", 'z', 5u);
        assert ~[~""] == splitn_char(~"", 'z', 5u);
        assert ~[~"ok"] == splitn_char(~"ok", 'z', 5u);
    }


    #[test]
    fn test_splitn_char_3() {
        let data = ~"ประเทศไทย中华Việt Nam";
        assert ~[~"ประเทศไทย中华", ~"iệt Nam"]
            == splitn_char(data, 'V', 1u);
        assert ~[~"ประเ", ~"ศไทย中华Việt Nam"]
            == splitn_char(data, 'ท', 1u);

    }

    #[test]
    fn test_split_str() {
        fn t(s: ~str, sep: &a/str, i: int, k: ~str) {
            let v = split_str(s, sep);
            assert v[i] == k;
        }

        t(~"--1233345--", ~"12345", 0, ~"--1233345--");
        t(~"abc::hello::there", ~"::", 0, ~"abc");
        t(~"abc::hello::there", ~"::", 1, ~"hello");
        t(~"abc::hello::there", ~"::", 2, ~"there");
        t(~"::hello::there", ~"::", 0, ~"");
        t(~"hello::there::", ~"::", 2, ~"");
        t(~"::hello::there::", ~"::", 3, ~"");

        let data = ~"ประเทศไทย中华Việt Nam";
        assert ~[~"ประเทศไทย", ~"Việt Nam"]
            == split_str (data, ~"中华");

        assert ~[~"", ~"XXX", ~"YYY", ~""]
            == split_str(~"zzXXXzzYYYzz", ~"zz");

        assert ~[~"zz", ~"zYYYz"]
            == split_str(~"zzXXXzYYYz", ~"XXX");


        assert ~[~"", ~"XXX", ~"YYY", ~""] == split_str(~".XXX.YYY.", ~".");
        assert ~[~""] == split_str(~"", ~".");
        assert ~[~"",~""] == split_str(~"zz", ~"zz");
        assert ~[~"ok"] == split_str(~"ok", ~"z");
        assert ~[~"",~"z"] == split_str(~"zzz", ~"zz");
        assert ~[~"",~"",~"z"] == split_str(~"zzzzz", ~"zz");
    }


    #[test]
    fn test_split() {
        let data = ~"ประเทศไทย中华Việt Nam";
        assert ~[~"ประเทศไทย中", ~"Việt Nam"]
            == split (data, |cc| cc == '华');

        assert ~[~"", ~"", ~"XXX", ~"YYY", ~""]
            == split(~"zzXXXzYYYz", char::is_lowercase);

        assert ~[~"zz", ~"", ~"", ~"z", ~"", ~"", ~"z"]
            == split(~"zzXXXzYYYz", char::is_uppercase);

        assert ~[~"",~""] == split(~"z", |cc| cc == 'z');
        assert ~[~""] == split(~"", |cc| cc == 'z');
        assert ~[~"ok"] == split(~"ok", |cc| cc == 'z');
    }

    #[test]
    fn test_lines() {
        let lf = ~"\nMary had a little lamb\nLittle lamb\n";
        let crlf = ~"\r\nMary had a little lamb\r\nLittle lamb\r\n";

        assert ~[~"", ~"Mary had a little lamb", ~"Little lamb", ~""]
            == lines(lf);

        assert ~[~"", ~"Mary had a little lamb", ~"Little lamb", ~""]
            == lines_any(lf);

        assert ~[~"\r", ~"Mary had a little lamb\r", ~"Little lamb\r", ~""]
            == lines(crlf);

        assert ~[~"", ~"Mary had a little lamb", ~"Little lamb", ~""]
            == lines_any(crlf);

        assert ~[~""] == lines    (~"");
        assert ~[~""] == lines_any(~"");
        assert ~[~"",~""] == lines    (~"\n");
        assert ~[~"",~""] == lines_any(~"\n");
        assert ~[~"banana"] == lines    (~"banana");
        assert ~[~"banana"] == lines_any(~"banana");
    }

    #[test]
    fn test_words () {
        let data = ~"\nMary had a little lamb\nLittle lamb\n";
        assert ~[~"Mary",~"had",~"a",~"little",~"lamb",~"Little",~"lamb"]
            == words(data);

        assert ~[~"ok"] == words(~"ok");
        assert ~[] == words(~"");
    }

    #[test]
    fn test_find_str() {
        // byte positions
        assert find_str(~"banana", ~"apple pie").is_none();
        assert find_str(~"", ~"") == Some(0u);

        let data = ~"ประเทศไทย中华Việt Nam";
        assert find_str(data, ~"")     == Some(0u);
        assert find_str(data, ~"ประเ") == Some( 0u);
        assert find_str(data, ~"ะเ")   == Some( 6u);
        assert find_str(data, ~"中华") == Some(27u);
        assert find_str(data, ~"ไท华").is_none();
    }

    #[test]
    fn test_find_str_between() {
        // byte positions
        assert find_str_between(~"", ~"", 0u, 0u) == Some(0u);

        let data = ~"abcabc";
        assert find_str_between(data, ~"ab", 0u, 6u) == Some(0u);
        assert find_str_between(data, ~"ab", 2u, 6u) == Some(3u);
        assert find_str_between(data, ~"ab", 2u, 4u).is_none();

        let mut data = ~"ประเทศไทย中华Việt Nam";
        data += data;
        assert find_str_between(data, ~"", 0u, 43u) == Some(0u);
        assert find_str_between(data, ~"", 6u, 43u) == Some(6u);

        assert find_str_between(data, ~"ประ", 0u, 43u) == Some( 0u);
        assert find_str_between(data, ~"ทศไ", 0u, 43u) == Some(12u);
        assert find_str_between(data, ~"ย中", 0u, 43u) == Some(24u);
        assert find_str_between(data, ~"iệt", 0u, 43u) == Some(34u);
        assert find_str_between(data, ~"Nam", 0u, 43u) == Some(40u);

        assert find_str_between(data, ~"ประ", 43u, 86u) == Some(43u);
        assert find_str_between(data, ~"ทศไ", 43u, 86u) == Some(55u);
        assert find_str_between(data, ~"ย中", 43u, 86u) == Some(67u);
        assert find_str_between(data, ~"iệt", 43u, 86u) == Some(77u);
        assert find_str_between(data, ~"Nam", 43u, 86u) == Some(83u);
    }

    #[test]
    fn test_substr() {
        fn t(a: ~str, b: ~str, start: int) {
            assert substr(a, start as uint, len(b)) == b;
        }
        t(~"hello", ~"llo", 2);
        t(~"hello", ~"el", 1);
        assert ~"ะเทศไท" == substr(~"ประเทศไทย中华Việt Nam", 6u, 6u);
    }

    #[test]
    fn test_concat() {
        fn t(v: ~[~str], s: ~str) { assert concat(v) == s; }
        t(~[~"you", ~"know", ~"I'm", ~"no", ~"good"], ~"youknowI'mnogood");
        let v: ~[~str] = ~[];
        t(v, ~"");
        t(~[~"hi"], ~"hi");
    }

    #[test]
    fn test_connect() {
        fn t(v: ~[~str], sep: ~str, s: ~str) {
            assert connect(v, sep) == s;
        }
        t(~[~"you", ~"know", ~"I'm", ~"no", ~"good"],
          ~" ", ~"you know I'm no good");
        let v: ~[~str] = ~[];
        t(v, ~" ", ~"");
        t(~[~"hi"], ~" ", ~"hi");
    }

    #[test]
    fn test_to_upper() {
        // libc::toupper, and hence str::to_upper
        // are culturally insensitive: they only work for ASCII
        // (see Issue #1347)
        let unicode = ~""; //"\u65e5\u672c"; // uncomment once non-ASCII works
        let input = ~"abcDEF" + unicode + ~"xyz:.;";
        let expected = ~"ABCDEF" + unicode + ~"XYZ:.;";
        let actual = to_upper(input);
        assert expected == actual;
    }

    #[test]
    fn test_to_lower() {
        assert ~"" == map(~"", |c| libc::tolower(c as c_char) as char);
        assert ~"ymca" == map(~"YMCA",
                             |c| libc::tolower(c as c_char) as char);
    }

    #[test]
    fn test_unsafe_slice() {
        unsafe {
            assert ~"ab" == unsafe::slice_bytes(~"abc", 0u, 2u);
            assert ~"bc" == unsafe::slice_bytes(~"abc", 1u, 3u);
            assert ~"" == unsafe::slice_bytes(~"abc", 1u, 1u);
            fn a_million_letter_a() -> ~str {
                let mut i = 0;
                let mut rs = ~"";
                while i < 100000 { push_str(rs, ~"aaaaaaaaaa"); i += 1; }
                return rs;
            }
            fn half_a_million_letter_a() -> ~str {
                let mut i = 0;
                let mut rs = ~"";
                while i < 100000 { push_str(rs, ~"aaaaa"); i += 1; }
                return rs;
            }
            assert half_a_million_letter_a() ==
                unsafe::slice_bytes(a_million_letter_a(), 0u, 500000u);
        }
    }

    #[test]
    fn test_starts_with() {
        assert (starts_with(~"", ~""));
        assert (starts_with(~"abc", ~""));
        assert (starts_with(~"abc", ~"a"));
        assert (!starts_with(~"a", ~"abc"));
        assert (!starts_with(~"", ~"abc"));
    }

    #[test]
    fn test_ends_with() {
        assert (ends_with(~"", ~""));
        assert (ends_with(~"abc", ~""));
        assert (ends_with(~"abc", ~"c"));
        assert (!ends_with(~"a", ~"abc"));
        assert (!ends_with(~"", ~"abc"));
    }

    #[test]
    fn test_is_empty() {
        assert (is_empty(~""));
        assert (!is_empty(~"a"));
    }

    #[test]
    fn test_is_not_empty() {
        assert (is_not_empty(~"a"));
        assert (!is_not_empty(~""));
    }

    #[test]
    fn test_replace() {
        let a = ~"a";
        assert replace(~"", a, ~"b") == ~"";
        assert replace(~"a", a, ~"b") == ~"b";
        assert replace(~"ab", a, ~"b") == ~"bb";
        let test = ~"test";
        assert replace(~" test test ", test, ~"toast") == ~" toast toast ";
        assert replace(~" test test ", test, ~"") == ~"   ";
    }

    #[test]
    fn test_replace_2a() {
        let data = ~"ประเทศไทย中华";
        let repl = ~"دولة الكويت";

        let a = ~"ประเ";
        let A = ~"دولة الكويتทศไทย中华";
        assert (replace(data, a, repl) ==  A);
    }

    #[test]
    fn test_replace_2b() {
        let data = ~"ประเทศไทย中华";
        let repl = ~"دولة الكويت";

        let b = ~"ะเ";
        let B = ~"ปรدولة الكويتทศไทย中华";
        assert (replace(data, b,   repl) ==  B);
    }

    #[test]
    fn test_replace_2c() {
        let data = ~"ประเทศไทย中华";
        let repl = ~"دولة الكويت";

        let c = ~"中华";
        let C = ~"ประเทศไทยدولة الكويت";
        assert (replace(data, c, repl) ==  C);
    }

    #[test]
    fn test_replace_2d() {
        let data = ~"ประเทศไทย中华";
        let repl = ~"دولة الكويت";

        let d = ~"ไท华";
        assert (replace(data, d, repl) == data);
    }

    #[test]
    fn test_slice() {
        assert ~"ab" == slice(~"abc", 0u, 2u);
        assert ~"bc" == slice(~"abc", 1u, 3u);
        assert ~"" == slice(~"abc", 1u, 1u);
        assert ~"\u65e5" == slice(~"\u65e5\u672c", 0u, 3u);

        let data = ~"ประเทศไทย中华";
        assert ~"ป" == slice(data, 0u, 3u);
        assert ~"ร" == slice(data, 3u, 6u);
        assert ~"" == slice(data, 3u, 3u);
        assert ~"华" == slice(data, 30u, 33u);

        fn a_million_letter_X() -> ~str {
            let mut i = 0;
            let mut rs = ~"";
            while i < 100000 { push_str(rs, ~"华华华华华华华华华华"); i += 1; }
            return rs;
        }
        fn half_a_million_letter_X() -> ~str {
            let mut i = 0;
            let mut rs = ~"";
            while i < 100000 { push_str(rs, ~"华华华华华"); i += 1; }
            return rs;
        }
        assert half_a_million_letter_X() ==
            slice(a_million_letter_X(), 0u, 3u * 500000u);
    }

    #[test]
    fn test_slice_2() {
        let ss = ~"中华Việt Nam";

        assert ~"华" == slice(ss, 3u, 6u);
        assert ~"Việt Nam" == slice(ss, 6u, 16u);

        assert ~"ab" == slice(~"abc", 0u, 2u);
        assert ~"bc" == slice(~"abc", 1u, 3u);
        assert ~"" == slice(~"abc", 1u, 1u);

        assert ~"中" == slice(ss, 0u, 3u);
        assert ~"华V" == slice(ss, 3u, 7u);
        assert ~"" == slice(ss, 3u, 3u);
        /*0: 中
          3: 华
          6: V
          7: i
          8: ệ
         11: t
         12:
         13: N
         14: a
         15: m */
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_slice_fail() {
        slice(~"中华Việt Nam", 0u, 2u);
    }

    #[test]
    fn test_trim_left_chars() {
        assert trim_left_chars(~" *** foo *** ", ~[]) == ~" *** foo *** ";
        assert trim_left_chars(~" *** foo *** ", ~['*', ' ']) == ~"foo *** ";
        assert trim_left_chars(~" ***  *** ", ~['*', ' ']) == ~"";
        assert trim_left_chars(~"foo *** ", ~['*', ' ']) == ~"foo *** ";
    }

    #[test]
    fn test_trim_right_chars() {
        assert trim_right_chars(~" *** foo *** ", ~[]) == ~" *** foo *** ";
        assert trim_right_chars(~" *** foo *** ", ~['*', ' ']) == ~" *** foo";
        assert trim_right_chars(~" ***  *** ", ~['*', ' ']) == ~"";
        assert trim_right_chars(~" *** foo", ~['*', ' ']) == ~" *** foo";
    }

    #[test]
    fn test_trim_chars() {
        assert trim_chars(~" *** foo *** ", ~[]) == ~" *** foo *** ";
        assert trim_chars(~" *** foo *** ", ~['*', ' ']) == ~"foo";
        assert trim_chars(~" ***  *** ", ~['*', ' ']) == ~"";
        assert trim_chars(~"foo", ~['*', ' ']) == ~"foo";
    }

    #[test]
    fn test_trim_left() {
        assert (trim_left(~"") == ~"");
        assert (trim_left(~"a") == ~"a");
        assert (trim_left(~"    ") == ~"");
        assert (trim_left(~"     blah") == ~"blah");
        assert (trim_left(~"   \u3000  wut") == ~"wut");
        assert (trim_left(~"hey ") == ~"hey ");
    }

    #[test]
    fn test_trim_right() {
        assert (trim_right(~"") == ~"");
        assert (trim_right(~"a") == ~"a");
        assert (trim_right(~"    ") == ~"");
        assert (trim_right(~"blah     ") == ~"blah");
        assert (trim_right(~"wut   \u3000  ") == ~"wut");
        assert (trim_right(~" hey") == ~" hey");
    }

    #[test]
    fn test_trim() {
        assert (trim(~"") == ~"");
        assert (trim(~"a") == ~"a");
        assert (trim(~"    ") == ~"");
        assert (trim(~"    blah     ") == ~"blah");
        assert (trim(~"\nwut   \u3000  ") == ~"wut");
        assert (trim(~" hey dude ") == ~"hey dude");
    }

    #[test]
    fn test_is_whitespace() {
        assert (is_whitespace(~""));
        assert (is_whitespace(~" "));
        assert (is_whitespace(~"\u2009")); // Thin space
        assert (is_whitespace(~"  \n\t   "));
        assert (!is_whitespace(~"   _   "));
    }

    #[test]
    fn test_is_ascii() {
        assert (is_ascii(~""));
        assert (is_ascii(~"a"));
        assert (!is_ascii(~"\u2009"));
    }

    #[test]
    fn test_shift_byte() {
        let mut s = ~"ABC";
        let b = unsafe { unsafe::shift_byte(s) };
        assert (s == ~"BC");
        assert (b == 65u8);
    }

    #[test]
    fn test_pop_byte() {
        let mut s = ~"ABC";
        let b = unsafe { unsafe::pop_byte(s) };
        assert (s == ~"AB");
        assert (b == 67u8);
    }

    #[test]
    fn test_unsafe_from_bytes() {
        let a = ~[65u8, 65u8, 65u8, 65u8, 65u8, 65u8, 65u8];
        let b = unsafe { unsafe::from_bytes(a) };
        assert (b == ~"AAAAAAA");
    }

    #[test]
    fn test_from_bytes() {
        let ss = ~"ศไทย中华Việt Nam";
        let bb = ~[0xe0_u8, 0xb8_u8, 0xa8_u8,
                  0xe0_u8, 0xb9_u8, 0x84_u8,
                  0xe0_u8, 0xb8_u8, 0x97_u8,
                  0xe0_u8, 0xb8_u8, 0xa2_u8,
                  0xe4_u8, 0xb8_u8, 0xad_u8,
                  0xe5_u8, 0x8d_u8, 0x8e_u8,
                  0x56_u8, 0x69_u8, 0xe1_u8,
                  0xbb_u8, 0x87_u8, 0x74_u8,
                  0x20_u8, 0x4e_u8, 0x61_u8,
                  0x6d_u8];

         assert ss == from_bytes(bb);
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_from_bytes_fail() {
        let bb = ~[0xff_u8, 0xb8_u8, 0xa8_u8,
                  0xe0_u8, 0xb9_u8, 0x84_u8,
                  0xe0_u8, 0xb8_u8, 0x97_u8,
                  0xe0_u8, 0xb8_u8, 0xa2_u8,
                  0xe4_u8, 0xb8_u8, 0xad_u8,
                  0xe5_u8, 0x8d_u8, 0x8e_u8,
                  0x56_u8, 0x69_u8, 0xe1_u8,
                  0xbb_u8, 0x87_u8, 0x74_u8,
                  0x20_u8, 0x4e_u8, 0x61_u8,
                  0x6d_u8];

         let _x = from_bytes(bb);
    }

    #[test]
    fn test_from_buf() {
        unsafe {
            let a = ~[65u8, 65u8, 65u8, 65u8, 65u8, 65u8, 65u8, 0u8];
            let b = vec::unsafe::to_ptr(a);
            let c = unsafe::from_buf(b);
            assert (c == ~"AAAAAAA");
        }
    }

    #[test]
    #[ignore(cfg(windows))]
    #[should_fail]
    fn test_as_bytes_fail() {
        // Don't double free
        as_bytes::<()>(~"", |_bytes| fail );
    }

    #[test]
    fn test_as_buf() {
        let a = ~"Abcdefg";
        let b = as_buf(a, |buf, _l| {
            assert unsafe { *buf } == 65u8;
            100
        });
        assert (b == 100);
    }

    #[test]
    fn test_as_buf_small() {
        let a = ~"A";
        let b = as_buf(a, |buf, _l| {
            assert unsafe { *buf } == 65u8;
            100
        });
        assert (b == 100);
    }

    #[test]
    fn test_as_buf2() {
        unsafe {
            let s = ~"hello";
            let sb = as_buf(s, |b, _l| b);
            let s_cstr = unsafe::from_buf(sb);
            assert s_cstr == s;
        }
    }

    #[test]
    fn test_as_buf_3() {
        let a = ~"hello";
        do as_buf(a) |buf, len| {
            unsafe {
                assert a[0] == 'h' as u8;
                assert *buf == 'h' as u8;
                assert len == 6u;
                assert *ptr::offset(buf,4u) == 'o' as u8;
                assert *ptr::offset(buf,5u) == 0u8;
            }
        }
    }

    #[test]
    fn vec_str_conversions() {
        let s1: ~str = ~"All mimsy were the borogoves";

        let v: ~[u8] = to_bytes(s1);
        let s2: ~str = from_bytes(v);
        let mut i: uint = 0u;
        let n1: uint = len(s1);
        let n2: uint = vec::len::<u8>(v);
        assert (n1 == n2);
        while i < n1 {
            let a: u8 = s1[i];
            let b: u8 = s2[i];
            log(debug, a);
            log(debug, b);
            assert (a == b);
            i += 1u;
        }
    }

    #[test]
    fn test_contains() {
        assert contains(~"abcde", ~"bcd");
        assert contains(~"abcde", ~"abcd");
        assert contains(~"abcde", ~"bcde");
        assert contains(~"abcde", ~"");
        assert contains(~"", ~"");
        assert !contains(~"abcde", ~"def");
        assert !contains(~"", ~"a");

        let data = ~"ประเทศไทย中华Việt Nam";
        assert  contains(data, ~"ประเ");
        assert  contains(data, ~"ะเ");
        assert  contains(data, ~"中华");
        assert !contains(data, ~"ไท华");
    }

    #[test]
    fn test_contains_char() {
        assert contains_char(~"abc", 'b');
        assert contains_char(~"a", 'a');
        assert !contains_char(~"abc", 'd');
        assert !contains_char(~"", 'a');
    }

    #[test]
    fn test_chars_iter() {
        let mut i = 0;
        do chars_iter(~"x\u03c0y") |ch| {
            match i {
              0 => assert ch == 'x',
              1 => assert ch == '\u03c0',
              2 => assert ch == 'y',
              _ => fail ~"test_chars_iter failed"
            }
            i += 1;
        }

        chars_iter(~"", |_ch| fail ); // should not fail
    }

    #[test]
    fn test_bytes_iter() {
        let mut i = 0;

        do bytes_iter(~"xyz") |bb| {
            match i {
              0 => assert bb == 'x' as u8,
              1 => assert bb == 'y' as u8,
              2 => assert bb == 'z' as u8,
              _ => fail ~"test_bytes_iter failed"
            }
            i += 1;
        }

        bytes_iter(~"", |bb| assert bb == 0u8);
    }

    #[test]
    fn test_split_char_iter() {
        let data = ~"\nMary had a little lamb\nLittle lamb\n";

        let mut ii = 0;

        do split_char_iter(data, ' ') |xx| {
            match ii {
              0 => assert ~"\nMary" == xx,
              1 => assert ~"had"    == xx,
              2 => assert ~"a"      == xx,
              3 => assert ~"little" == xx,
              _ => ()
            }
            ii += 1;
        }
    }

    #[test]
    fn test_splitn_char_iter() {
        let data = ~"\nMary had a little lamb\nLittle lamb\n";

        let mut ii = 0;

        do splitn_char_iter(data, ' ', 2u) |xx| {
            match ii {
              0 => assert ~"\nMary" == xx,
              1 => assert ~"had"    == xx,
              2 => assert ~"a little lamb\nLittle lamb\n" == xx,
              _ => ()
            }
            ii += 1;
        }
    }

    #[test]
    fn test_words_iter() {
        let data = ~"\nMary had a little lamb\nLittle lamb\n";

        let mut ii = 0;

        do words_iter(data) |ww| {
            match ii {
              0 => assert ~"Mary"   == ww,
              1 => assert ~"had"    == ww,
              2 => assert ~"a"      == ww,
              3 => assert ~"little" == ww,
              _ => ()
            }
            ii += 1;
        }

        words_iter(~"", |_x| fail); // should not fail
    }

    #[test]
    fn test_lines_iter () {
        let lf = ~"\nMary had a little lamb\nLittle lamb\n";

        let mut ii = 0;

        do lines_iter(lf) |x| {
            match ii {
                0 => assert ~"" == x,
                1 => assert ~"Mary had a little lamb" == x,
                2 => assert ~"Little lamb" == x,
                3 => assert ~"" == x,
                _ => ()
            }
            ii += 1;
        }
    }

    #[test]
    fn test_map() {
        assert ~"" == map(~"", |c| libc::toupper(c as c_char) as char);
        assert ~"YMCA" == map(~"ymca",
                              |c| libc::toupper(c as c_char) as char);
    }

    #[test]
    fn test_all() {
        assert true  == all(~"", char::is_uppercase);
        assert false == all(~"ymca", char::is_uppercase);
        assert true  == all(~"YMCA", char::is_uppercase);
        assert false == all(~"yMCA", char::is_uppercase);
        assert false == all(~"YMCy", char::is_uppercase);
    }

    #[test]
    fn test_any() {
        assert false  == any(~"", char::is_uppercase);
        assert false == any(~"ymca", char::is_uppercase);
        assert true  == any(~"YMCA", char::is_uppercase);
        assert true == any(~"yMCA", char::is_uppercase);
        assert true == any(~"Ymcy", char::is_uppercase);
    }

    #[test]
    fn test_chars() {
        let ss = ~"ศไทย中华Việt Nam";
        assert ~['ศ','ไ','ท','ย','中','华','V','i','ệ','t',' ','N','a','m']
            == chars(ss);
    }

    #[test]
    fn test_utf16() {
        let pairs =
            ~[(~"𐍅𐌿𐌻𐍆𐌹𐌻𐌰\n",
              ~[0xd800_u16, 0xdf45_u16, 0xd800_u16, 0xdf3f_u16,
               0xd800_u16, 0xdf3b_u16, 0xd800_u16, 0xdf46_u16,
               0xd800_u16, 0xdf39_u16, 0xd800_u16, 0xdf3b_u16,
               0xd800_u16, 0xdf30_u16, 0x000a_u16]),

             (~"𐐒𐑉𐐮𐑀𐐲𐑋 𐐏𐐲𐑍\n",
              ~[0xd801_u16, 0xdc12_u16, 0xd801_u16,
               0xdc49_u16, 0xd801_u16, 0xdc2e_u16, 0xd801_u16,
               0xdc40_u16, 0xd801_u16, 0xdc32_u16, 0xd801_u16,
               0xdc4b_u16, 0x0020_u16, 0xd801_u16, 0xdc0f_u16,
               0xd801_u16, 0xdc32_u16, 0xd801_u16, 0xdc4d_u16,
               0x000a_u16]),

             (~"𐌀𐌖𐌋𐌄𐌑𐌉·𐌌𐌄𐌕𐌄𐌋𐌉𐌑\n",
              ~[0xd800_u16, 0xdf00_u16, 0xd800_u16, 0xdf16_u16,
               0xd800_u16, 0xdf0b_u16, 0xd800_u16, 0xdf04_u16,
               0xd800_u16, 0xdf11_u16, 0xd800_u16, 0xdf09_u16,
               0x00b7_u16, 0xd800_u16, 0xdf0c_u16, 0xd800_u16,
               0xdf04_u16, 0xd800_u16, 0xdf15_u16, 0xd800_u16,
               0xdf04_u16, 0xd800_u16, 0xdf0b_u16, 0xd800_u16,
               0xdf09_u16, 0xd800_u16, 0xdf11_u16, 0x000a_u16 ]),

             (~"𐒋𐒘𐒈𐒑𐒛𐒒 𐒕𐒓 𐒈𐒚𐒍 𐒏𐒜𐒒𐒖𐒆 𐒕𐒆\n",
              ~[0xd801_u16, 0xdc8b_u16, 0xd801_u16, 0xdc98_u16,
               0xd801_u16, 0xdc88_u16, 0xd801_u16, 0xdc91_u16,
               0xd801_u16, 0xdc9b_u16, 0xd801_u16, 0xdc92_u16,
               0x0020_u16, 0xd801_u16, 0xdc95_u16, 0xd801_u16,
               0xdc93_u16, 0x0020_u16, 0xd801_u16, 0xdc88_u16,
               0xd801_u16, 0xdc9a_u16, 0xd801_u16, 0xdc8d_u16,
               0x0020_u16, 0xd801_u16, 0xdc8f_u16, 0xd801_u16,
               0xdc9c_u16, 0xd801_u16, 0xdc92_u16, 0xd801_u16,
               0xdc96_u16, 0xd801_u16, 0xdc86_u16, 0x0020_u16,
               0xd801_u16, 0xdc95_u16, 0xd801_u16, 0xdc86_u16,
               0x000a_u16 ]) ];

        for vec::each(pairs) |p| {
            let (s, u) = copy p;
            assert to_utf16(s) == u;
            assert from_utf16(u) == s;
            assert from_utf16(to_utf16(s)) == s;
            assert to_utf16(from_utf16(u)) == u;
        }
    }

    #[test]
    fn test_each_char() {
        let s = ~"abc";
        let mut found_b = false;
        for each_char(s) |ch| {
            if ch == 'b' {
                found_b = true;
                break;
            }
        }
        assert found_b;
    }

    #[test]
    fn test_escape_unicode() {
        assert escape_unicode(~"abc") == ~"\\x61\\x62\\x63";
        assert escape_unicode(~"a c") == ~"\\x61\\x20\\x63";
        assert escape_unicode(~"\r\n\t") == ~"\\x0d\\x0a\\x09";
        assert escape_unicode(~"'\"\\") == ~"\\x27\\x22\\x5c";
        assert escape_unicode(~"\x00\x01\xfe\xff") == ~"\\x00\\x01\\xfe\\xff";
        assert escape_unicode(~"\u0100\uffff") == ~"\\u0100\\uffff";
        assert escape_unicode(~"\U00010000\U0010ffff") ==
            ~"\\U00010000\\U0010ffff";
        assert escape_unicode(~"ab\ufb00") == ~"\\x61\\x62\\ufb00";
        assert escape_unicode(~"\U0001d4ea\r") == ~"\\U0001d4ea\\x0d";
    }

    #[test]
    fn test_escape_default() {
        assert escape_default(~"abc") == ~"abc";
        assert escape_default(~"a c") == ~"a c";
        assert escape_default(~"\r\n\t") == ~"\\r\\n\\t";
        assert escape_default(~"'\"\\") == ~"\\'\\\"\\\\";
        assert escape_default(~"\u0100\uffff") == ~"\\u0100\\uffff";
        assert escape_default(~"\U00010000\U0010ffff") ==
            ~"\\U00010000\\U0010ffff";
        assert escape_default(~"ab\ufb00") == ~"ab\\ufb00";
        assert escape_default(~"\U0001d4ea\r") == ~"\\U0001d4ea\\r";
    }

}
