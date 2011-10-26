/*
Module: str

String manipulation.
*/

export eq, lteq, hash, is_empty, is_not_empty, is_whitespace, byte_len, index,
       rindex, find, starts_with, ends_with, substr, slice, split, concat,
       connect, to_upper, replace, char_slice, trim_left, trim_right, trim,
       unshift_char, shift_char, pop_char, push_char, is_utf8, from_chars,
       to_chars, char_len, char_at, bytes, is_ascii, shift_byte, pop_byte,
       unsafe_from_byte, unsafe_from_bytes, from_char, char_range_at,
       str_from_cstr, sbuf, as_buf, push_byte, utf8_char_width, safe_slice,
       contains;

native "c-stack-cdecl" mod rustrt {
    fn rust_str_push(&s: str, ch: u8);
}

/*
Function: eq

Bytewise string equality
*/
fn eq(&&a: str, &&b: str) -> bool { a == b }

/*
Function: lteq

Bytewise less than or equal
*/
fn lteq(&&a: str, &&b: str) -> bool { a <= b }

/*
Function: hash

String hash function
*/
fn hash(&&s: str) -> uint {
    // djb hash.
    // FIXME: replace with murmur.

    let u: uint = 5381u;
    for c: u8 in s { u *= 33u; u += c as uint; }
    ret u;
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

/*
Function: is_utf8

Determines if a vector uf bytes contains valid UTF-8
*/
fn is_utf8(v: [u8]) -> bool {
    let i = 0u;
    let total = vec::len::<u8>(v);
    while i < total {
        let chsize = utf8_char_width(v[i]);
        if chsize == 0u { ret false; }
        if i + chsize > total { ret false; }
        i += 1u;
        while chsize > 1u {
            if v[i] & 192u8 != tag_cont_u8 { ret false; }
            i += 1u;
            chsize -= 1u;
        }
    }
    ret true;
}

/*
Function: is_ascii

Determines if a string contains only ASCII characters
*/
fn is_ascii(s: str) -> bool {
    let i: uint = byte_len(s);
    while i > 0u { i -= 1u; if s[i] & 128u8 != 0u8 { ret false; } }
    ret true;
}

/*
Predicate: is_empty

Returns true if the string has length 0
*/
pure fn is_empty(s: str) -> bool { for c: u8 in s { ret false; } ret true; }

/*
Predicate: is_not_empty

Returns true if the string has length greater than 0
*/
pure fn is_not_empty(s: str) -> bool { !is_empty(s) }

/*
Function: is_whitespace

Returns true if the string contains only whitespace
*/
fn is_whitespace(s: str) -> bool {
    let i = 0u;
    let len = char_len(s);
    while i < len {
        // FIXME: This is not how char_at works
        if !char::is_whitespace(char_at(s, i)) { ret false; }
        i += 1u;
    }
    ret true;
}

/*
Function: byte_len

Returns the length in bytes of a string
*/
fn byte_len(s: str) -> uint {
    let v: [u8] = unsafe::reinterpret_cast(s);
    let vlen = vec::len(v);
    unsafe::leak(v);
    // There should always be a null terminator
    assert (vlen > 0u);
    ret vlen - 1u;
}

/*
Function: bytes

Converts a string to a vector of bytes
*/
fn bytes(s: str) -> [u8] {
    let v = unsafe::reinterpret_cast(s);
    let vcopy = vec::slice(v, 0u, vec::len(v) - 1u);
    unsafe::leak(v);
    ret vcopy;
}

/*
Function: unsafe_from_bytes

Converts a vector of bytes to a string. Does not verify that the
vector contains valid UTF-8.
*/
fn unsafe_from_bytes(v: [mutable? u8]) -> str {
    let vcopy: [u8] = v + [0u8];
    let scopy: str = unsafe::reinterpret_cast(vcopy);
    unsafe::leak(vcopy);
    ret scopy;
}

/*
Function: unsafe_from_byte

Converts a byte to a string. Does not verify that the byte is
valid UTF-8.
*/
fn unsafe_from_byte(u: u8) -> str { unsafe_from_bytes([u]) }

fn push_utf8_bytes(&s: str, ch: char) {
    let code = ch as uint;
    let bytes =
        if code < max_one_b {
            [code as u8]
        } else if code < max_two_b {
            [code >> 6u & 31u | tag_two_b as u8, code & 63u | tag_cont as u8]
        } else if code < max_three_b {
            [code >> 12u & 15u | tag_three_b as u8,
             code >> 6u & 63u | tag_cont as u8, code & 63u | tag_cont as u8]
        } else if code < max_four_b {
            [code >> 18u & 7u | tag_four_b as u8,
             code >> 12u & 63u | tag_cont as u8,
             code >> 6u & 63u | tag_cont as u8, code & 63u | tag_cont as u8]
        } else if code < max_five_b {
            [code >> 24u & 3u | tag_five_b as u8,
             code >> 18u & 63u | tag_cont as u8,
             code >> 12u & 63u | tag_cont as u8,
             code >> 6u & 63u | tag_cont as u8, code & 63u | tag_cont as u8]
        } else {
            [code >> 30u & 1u | tag_six_b as u8,
             code >> 24u & 63u | tag_cont as u8,
             code >> 18u & 63u | tag_cont as u8,
             code >> 12u & 63u | tag_cont as u8,
             code >> 6u & 63u | tag_cont as u8, code & 63u | tag_cont as u8]
        };
    push_bytes(s, bytes);
}

/*
Function: from_char

Convert a char to a string
*/
fn from_char(ch: char) -> str {
    let buf = "";
    push_utf8_bytes(buf, ch);
    ret buf;
}

/*
Function: from_chars

Convert a vector of chars to a string
*/
fn from_chars(chs: [char]) -> str {
    let buf = "";
    for ch: char in chs { push_utf8_bytes(buf, ch); }
    ret buf;
}

/*
Function: utf8_char_width

FIXME: What does this function do?
*/
fn utf8_char_width(b: u8) -> uint {
    let byte: uint = b as uint;
    if byte < 128u { ret 1u; }
    if byte < 192u {
        ret 0u; // Not a valid start byte

    }
    if byte < 224u { ret 2u; }
    if byte < 240u { ret 3u; }
    if byte < 248u { ret 4u; }
    if byte < 252u { ret 5u; }
    ret 6u;
}

/*
Function: char_range_at

Pluck a character out of a string and return the index of the next character.
This function can be used to iterate over the unicode characters of a string.

Example:

> let s = "Clam chowder, hot sauce, pork rinds";
> let i = 0;
> while i < len(s) {
>   let {ch, next} = char_range_at(s, i);
>   log ch;
>   i = next;
> }

Parameters:

s - The string
i - The byte offset of the char to extract

Returns:

A record {ch: char, next: uint} containing the char value and the byte
index of the next unicode character.

Failure:

If `i` is greater than or equal to the length of the string.
If `i` is not the index of the beginning of a valid UTF-8 character.
*/
fn char_range_at(s: str, i: uint) -> {ch: char, next: uint} {
    let b0 = s[i];
    let w = utf8_char_width(b0);
    assert (w != 0u);
    if w == 1u { ret {ch: b0 as char, next: i + 1u}; }
    let val = 0u;
    let end = i + w;
    i += 1u;
    while i < end {
        let byte = s[i];
        assert (byte & 192u8 == tag_cont_u8);
        val <<= 6u;
        val += byte & 63u8 as uint;
        i += 1u;
    }
    // Clunky way to get the right bits from the first byte. Uses two shifts,
    // the first to clip off the marker bits at the left of the byte, and then
    // a second (as uint) to get it to the right position.
    val += (b0 << (w + 1u as u8) as uint) << (w - 1u) * 6u - w - 1u;
    ret {ch: val as char, next: i};
}

/*
Function: char_at

Pluck a character out of a string
*/
fn char_at(s: str, i: uint) -> char { ret char_range_at(s, i).ch; }

/*
Function: char_len

Count the number of unicode characters in a string
*/
fn char_len(s: str) -> uint {
    let i = 0u;
    let len = 0u;
    let total = byte_len(s);
    while i < total {
        let chsize = utf8_char_width(s[i]);
        assert (chsize > 0u);
        len += 1u;
        i += chsize;
    }
    assert (i == total);
    ret len;
}

/*
Function: to_chars

Convert a string to a vector of characters
*/
fn to_chars(s: str) -> [char] {
    let buf: [char] = [];
    let i = 0u;
    let len = byte_len(s);
    while i < len {
        let cur = char_range_at(s, i);
        buf += [cur.ch];
        i = cur.next;
    }
    ret buf;
}

/*
Function: push_char

Append a character to a string
*/
fn push_char(&s: str, ch: char) { s += from_char(ch); }

/*
Function: pop_char

Remove the final character from a string and return it.

Failure:

If the string does not contain any characters.
*/
fn pop_char(&s: str) -> char {
    let end = byte_len(s);
    while end > 0u && s[end - 1u] & 192u8 == tag_cont_u8 { end -= 1u; }
    assert (end > 0u);
    let ch = char_at(s, end - 1u);
    s = substr(s, 0u, end - 1u);
    ret ch;
}

/*
Function: shift_char

Remove the first character from a string and return it.

Failure:

If the string does not contain any characters.
*/
fn shift_char(&s: str) -> char {
    let r = char_range_at(s, 0u);
    s = substr(s, r.next, byte_len(s) - r.next);
    ret r.ch;
}

/*
Function: unshift_char

Prepend a char to a string
*/
fn unshift_char(&s: str, ch: char) { s = from_char(ch) + s; }

/*
Function: index

Returns the index of the first matching byte. Returns -1 if
no match is found.
*/
fn index(s: str, c: u8) -> int {
    let i: int = 0;
    for k: u8 in s { if k == c { ret i; } i += 1; }
    ret -1;
}

/*
Function: rindex

Returns the index of the last matching byte. Returns -1
if no match is found.
*/
fn rindex(s: str, c: u8) -> int {
    let n: int = byte_len(s) as int;
    while n >= 0 { if s[n] == c { ret n; } n -= 1; }
    ret n;
}

/*
Function: find

Finds the index of the first matching substring.
Returns -1 if `haystack` does not contain `needle`.

Parameters:

haystack - The string to look in
needle - The string to look for

Returns:

The index of the first occurance of `needle`, or -1 if not found.
*/
fn find(haystack: str, needle: str) -> int {
    let haystack_len: int = byte_len(haystack) as int;
    let needle_len: int = byte_len(needle) as int;
    if needle_len == 0 { ret 0; }
    fn match_at(haystack: str, needle: str, i: int) -> bool {
        let j: int = i;
        for c: u8 in needle { if haystack[j] != c { ret false; } j += 1; }
        ret true;
    }
    let i: int = 0;
    while i <= haystack_len - needle_len {
        if match_at(haystack, needle, i) { ret i; }
        i += 1;
    }
    ret -1;
}

/*
Function: contains

Returns true if one string contains another

Parameters:

haystack - The string to look in
needle - The string to look for
*/
fn contains(haystack: str, needle: str) -> bool {
    0 <= find(haystack, needle)
}

/*
Function: starts_with

Returns true if one string starts with another

Parameters:

haystack - The string to look in
needle - The string to look for
*/
fn starts_with(haystack: str, needle: str) -> bool {
    let haystack_len: uint = byte_len(haystack);
    let needle_len: uint = byte_len(needle);
    if needle_len == 0u { ret true; }
    if needle_len > haystack_len { ret false; }
    ret eq(substr(haystack, 0u, needle_len), needle);
}

/*
Function: ends_with

Returns true if one string ends with another

haystack - The string to look in
needle - The string to look for
*/
fn ends_with(haystack: str, needle: str) -> bool {
    let haystack_len: uint = byte_len(haystack);
    let needle_len: uint = byte_len(needle);
    ret if needle_len == 0u {
            true
        } else if needle_len > haystack_len {
            false
        } else {
            eq(substr(haystack, haystack_len - needle_len, needle_len),
               needle)
        };
}

/*
Function: substr

Take a substring of another. Returns a string containing `len` bytes
starting at byte offset `begin`.

This function is not unicode-safe.

Failure:

If `begin` + `len` is is greater than the byte length of the string
*/
fn substr(s: str, begin: uint, len: uint) -> str {
    ret slice(s, begin, begin + len);
}

/*
Function: slice

Takes a bytewise slice from a string. Returns the substring from
[`begin`..`end`).

This function is not unicode-safe.

Failure:

- If begin is greater than end.
- If end is greater than the length of the string.
*/
fn slice(s: str, begin: uint, end: uint) -> str {
    // FIXME: Typestate precondition
    assert (begin <= end);
    assert (end <= byte_len(s));

    let v: [u8] = unsafe::reinterpret_cast(s);
    let v2 = vec::slice(v, begin, end);
    unsafe::leak(v);
    v2 += [0u8];
    let s2: str = unsafe::reinterpret_cast(v2);
    unsafe::leak(v2);
    ret s2;
}

/*
Function: safe_slice
*/
fn safe_slice(s: str, begin: uint, end: uint) : uint::le(begin, end) -> str {
    // would need some magic to make this a precondition
    assert (end <= byte_len(s));
    ret slice(s, begin, end);
}

/*
Function: shift_byte

Removes the first byte from a string and returns it.

This function is not unicode-safe.
*/
fn shift_byte(&s: str) -> u8 {
    let len = byte_len(s);
    assert (len > 0u);
    let b = s[0];
    s = substr(s, 1u, len - 1u);
    ret b;
}

/*
Function: pop_byte

Removes the last byte from a string and returns it.

This function is not unicode-safe.
*/
fn pop_byte(&s: str) -> u8 {
    let len = byte_len(s);
    assert (len > 0u);
    let b = s[len - 1u];
    s = substr(s, 0u, len - 1u);
    ret b;
}

/*
Function: push_byte

Appends a byte to a string.

This function is not unicode-safe.
*/
fn push_byte(&s: str, b: u8) { rustrt::rust_str_push(s, b); }

/*
Function: push_bytes

Appends a vector of bytes to a string.

This function is not unicode-safe.
*/
fn push_bytes(&s: str, bytes: [u8]) {
    for byte in bytes { rustrt::rust_str_push(s, byte); }
}

/*
Function: split

Split a string at each occurance of a given separator

Returns:

A vector containing all the strings between each occurance of the separator
*/
fn split(s: str, sep: u8) -> [str] {
    let v: [str] = [];
    let accum: str = "";
    let ends_with_sep: bool = false;
    for c: u8 in s {
        if c == sep {
            v += [accum];
            accum = "";
            ends_with_sep = true;
        } else { accum += unsafe_from_byte(c); ends_with_sep = false; }
    }
    if byte_len(accum) != 0u || ends_with_sep { v += [accum]; }
    ret v;
}

/*
Function: concat

Concatenate a vector of strings
*/
fn concat(v: [str]) -> str {
    let s: str = "";
    for ss: str in v { s += ss; }
    ret s;
}

/*
Function: connect

Concatenate a vector of strings, placing a given separator between each
*/
fn connect(v: [str], sep: str) -> str {
    let s: str = "";
    let first: bool = true;
    for ss: str in v {
        if first { first = false; } else { s += sep; }
        s += ss;
    }
    ret s;
}

// FIXME: This only handles ASCII
/*
Function: to_upper

Convert a string to uppercase
*/
fn to_upper(s: str) -> str {
    let outstr = "";
    let ascii_a = 'a' as u8;
    let ascii_z = 'z' as u8;
    let diff = 32u8;
    for byte: u8 in s {
        let next;
        if ascii_a <= byte && byte <= ascii_z {
            next = byte - diff;
        } else { next = byte; }
        push_byte(outstr, next);
    }
    ret outstr;
}

// FIXME: This is super-inefficient
/*
Function: replace

Replace all occurances of one string with another

Parameters:

s - The string containing substrings to replace
from - The string to replace
to - The replacement string

Returns:

The original string with all occurances of `from` replaced with `to`
*/
fn replace(s: str, from: str, to: str) : is_not_empty(from) -> str {
    // FIXME (694): Shouldn't have to check this
    check (is_not_empty(from));
    if byte_len(s) == 0u {
        ret "";
    } else if starts_with(s, from) {
        ret to + replace(slice(s, byte_len(from), byte_len(s)), from, to);
    } else {
        ret unsafe_from_byte(s[0]) +
                replace(slice(s, 1u, byte_len(s)), from, to);
    }
}

// FIXME: Also not efficient
/*
Function: char_slice

Unicode-safe slice. Returns a slice of the given string containing
the characters in the range [`begin`..`end`). `begin` and `end` are
character indexes, not byte indexes.

Failure:

- If begin is greater than end
- If end is greater than the character length of the string
*/
fn char_slice(s: str, begin: uint, end: uint) -> str {
    from_chars(vec::slice(to_chars(s), begin, end))
}

/*
Function: trim_left

Returns a string with leading whitespace removed.
*/
fn trim_left(s: str) -> str {
    fn count_whities(s: [char]) -> uint {
        let i = 0u;
        while i < vec::len(s) {
            if !char::is_whitespace(s[i]) { break; }
            i += 1u;
        }
        ret i;
    }
    let chars = to_chars(s);
    let whities = count_whities(chars);
    ret from_chars(vec::slice(chars, whities, vec::len(chars)));
}

/*
Function: trim_right

Returns a string with trailing whitespace removed.
*/
fn trim_right(s: str) -> str {
    fn count_whities(s: [char]) -> uint {
        let i = vec::len(s);
        while 0u < i {
            if !char::is_whitespace(s[i - 1u]) { break; }
            i -= 1u;
        }
        ret i;
    }
    let chars = to_chars(s);
    let whities = count_whities(chars);
    ret from_chars(vec::slice(chars, 0u, whities));
}

/*
Function: trim

Returns a string with leading and trailing whitespace removed
*/
fn trim(s: str) -> str { trim_left(trim_right(s)) }

/*
Type: sbuf

An unsafe buffer of bytes. Corresponds to a C char pointer.
*/
type sbuf = *u8;

// NB: This is intentionally unexported because it's easy to misuse (there's
// no guarantee that the string is rooted). Instead, use as_buf below.
unsafe fn buf(s: str) -> sbuf {
    let saddr = ptr::addr_of(s);
    let vaddr: *[u8] = unsafe::reinterpret_cast(saddr);
    let buf = vec::to_ptr(*vaddr);
    ret buf;
}

/*
Function: as_buf

Work with the byte buffer of a string. Allows for unsafe manipulation
of strings, which is useful for native interop.

Example:

> let s = str::as_buf("PATH", { |path_buf| libc::getenv(path_buf) });

*/
fn as_buf<T>(s: str, f: block(sbuf) -> T) -> T unsafe {
    let buf = buf(s); f(buf)
}

/*
Function: str_from_cstr

Create a Rust string from a null-terminated C string
*/
unsafe fn str_from_cstr(cstr: sbuf) -> str {
    let res = "";
    let start = cstr;
    let curr = start;
    let i = 0u;
    while *curr != 0u8 {
        push_byte(res, *curr);
        i += 1u;
        curr = ptr::offset(start, i);
    }
    ret res;
}
