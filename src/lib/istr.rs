export eq, lteq, hash, is_empty, is_not_empty, is_whitespace, byte_len,
index, rindex, find, starts_with, ends_with, substr, slice, split,
concat, connect, to_upper, replace, char_slice, trim_left, trim_right, trim,
unshift_char, shift_char, pop_char, push_char, is_utf8, from_chars, to_chars,
char_len, char_at, bytes, is_ascii, shift_byte, pop_byte, unsafe_from_bytes;

export from_estr, to_estr;

fn from_estr(s: &str) -> istr {
    let s2 = ~"";
    for u in s {
        push_byte(s2, u);
    }
    ret s2;
}

fn to_estr(s: &istr) -> str {
    let s2 = "";
    for u in s {
        str::push_byte(s2, u);
    }
    ret s2;
}

fn eq(a: &istr, b: &istr) -> bool { a == b }

fn lteq(a: &istr, b: &istr) -> bool { a <= b }

fn hash(s: &istr) -> uint {
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

fn is_utf8(v: &[u8]) -> bool {
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

fn is_ascii(s: &istr) -> bool {
    let i: uint = byte_len(s);
    while i > 0u { i -= 1u; if s[i] & 128u8 != 0u8 { ret false; } }
    ret true;
}

/// Returns true if the string has length 0
pure fn is_empty(s: &istr) -> bool {
    for c: u8 in s { ret false; } ret true;
}

/// Returns true if the string has length greater than 0
pure fn is_not_empty(s: &istr) -> bool {
    !is_empty(s)
}

fn is_whitespace(s: &istr) -> bool {
    let i = 0u;
    let len = char_len(s);
    while i < len {
        if !char::is_whitespace(char_at(s, i)) { ret false; }
        i += 1u
    }
    ret true;
}

fn byte_len(s: &istr) -> uint {
    let v: [u8] = unsafe::reinterpret_cast(s);
    let vlen = vec::len(v);
    unsafe::leak(v);
    // There should always be a null terminator
    assert vlen > 0u;
    ret vlen - 1u;
}

fn bytes(s: &istr) -> [u8] {
    let v = unsafe::reinterpret_cast(s);
    let vcopy = vec::slice(v, 0u, vec::len(v) - 1u);
    unsafe::leak(v);
    ret vcopy;
}

fn unsafe_from_bytes(v: &[mutable? u8]) -> istr {
    let vcopy: [u8] = v + [0u8];
    let scopy: istr = unsafe::reinterpret_cast(vcopy);
    unsafe::leak(vcopy);
    ret scopy;
}

fn unsafe_from_byte(u: u8) -> istr {
    unsafe_from_bytes([u])
}

fn push_utf8_bytes(s: &mutable istr, ch: char) {
    let code = ch as uint;
    let bytes = if code < max_one_b {
        [code as u8]
    } else if code < max_two_b {
        [(code >> 6u & 31u | tag_two_b) as u8,
         (code & 63u | tag_cont) as u8]
    } else if code < max_three_b {
        [(code >> 12u & 15u | tag_three_b) as u8,
         (code >> 6u & 63u | tag_cont) as u8,
         (code & 63u | tag_cont) as u8]
    } else if code < max_four_b {
        [(code >> 18u & 7u | tag_four_b) as u8,
         (code >> 12u & 63u | tag_cont) as u8,
         (code >> 6u & 63u | tag_cont) as u8,
         (code & 63u | tag_cont) as u8]
    } else if code < max_five_b {
        [(code >> 24u & 3u | tag_five_b) as u8,
         (code >> 18u & 63u | tag_cont) as u8,
         (code >> 12u & 63u | tag_cont) as u8,
         (code >> 6u & 63u | tag_cont) as u8,
         (code & 63u | tag_cont) as u8]
    } else {
        [(code >> 30u & 1u | tag_six_b) as u8,
         (code >> 24u & 63u | tag_cont) as u8,
         (code >> 18u & 63u | tag_cont) as u8,
         (code >> 12u & 63u | tag_cont) as u8,
         (code >> 6u & 63u | tag_cont) as u8,
         (code & 63u | tag_cont) as u8]
    };
    push_bytes(s, bytes);
}

fn from_char(ch: char) -> istr {
    let buf = ~"";
    push_utf8_bytes(buf, ch);
    ret buf;
}

fn from_chars(chs: &[char]) -> istr {
    let buf = ~"";
    for ch: char in chs { push_utf8_bytes(buf, ch); }
    ret buf;
}

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

fn char_range_at(s: &istr, i: uint) -> {ch: char, next: uint} {
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

fn char_at(s: &istr, i: uint) -> char { ret char_range_at(s, i).ch; }

fn char_len(s: &istr) -> uint {
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

fn to_chars(s: &istr) -> [char] {
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

fn push_char(s: &mutable istr, ch: char) { s += from_char(ch); }

fn pop_char(s: &mutable istr) -> char {
    let end = byte_len(s);
    while end > 0u && s[end - 1u] & 192u8 == tag_cont_u8 { end -= 1u; }
    assert (end > 0u);
    let ch = char_at(s, end - 1u);
    s = substr(s, 0u, end - 1u);
    ret ch;
}

fn shift_char(s: &mutable istr) -> char {
    let r = char_range_at(s, 0u);
    s = substr(s, r.next, byte_len(s) - r.next);
    ret r.ch;
}

fn unshift_char(s: &mutable istr, ch: char) { s = from_char(ch) + s; }

fn index(s: &istr, c: u8) -> int {
    let i: int = 0;
    for k: u8 in s { if k == c { ret i; } i += 1; }
    ret -1;
}

fn rindex(s: &istr, c: u8) -> int {
    let n: int = byte_len(s) as int;
    while n >= 0 { if s[n] == c { ret n; } n -= 1; }
    ret n;
}

fn find(haystack: &istr, needle: &istr) -> int {
    let haystack_len: int = byte_len(haystack) as int;
    let needle_len: int = byte_len(needle) as int;
    if needle_len == 0 { ret 0; }
    fn match_at(haystack: &istr, needle: &istr, i: int) -> bool {
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

fn starts_with(haystack: &istr, needle: &istr) -> bool {
    let haystack_len: uint = byte_len(haystack);
    let needle_len: uint = byte_len(needle);
    if needle_len == 0u { ret true; }
    if needle_len > haystack_len { ret false; }
    ret eq(substr(haystack, 0u, needle_len), needle);
}

fn ends_with(haystack: &istr, needle: &istr) -> bool {
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

fn substr(s: &istr, begin: uint, len: uint) -> istr {
    ret slice(s, begin, begin + len);
}

fn slice(s: &istr, begin: uint, end: uint) -> istr {
    // FIXME: Typestate precondition
    assert (begin <= end);
    assert (end <= byte_len(s));

    let v: [u8] = unsafe::reinterpret_cast(s);
    let v2 = vec::slice(v, begin, end);
    unsafe::leak(v);
    v2 += [0u8];
    let s2: istr = unsafe::reinterpret_cast(v2);
    unsafe::leak(v2);
    ret s2;
}

fn safe_slice(s: &istr, begin: uint, end: uint)
    : uint::le(begin, end) -> istr {
    // would need some magic to make this a precondition
    assert (end <= byte_len(s));
    ret slice(s, begin, end);
}

fn shift_byte(s: &mutable istr) -> u8 {
    let len = byte_len(s);
    assert (len > 0u);
    let b = s[0];
    s = substr(s, 1u, len - 1u);
    ret b;
}

fn pop_byte(s: &mutable istr) -> u8 {
    let len = byte_len(s);
    assert (len > 0u);
    let b = s[len - 1u];
    s = substr(s, 0u, len - 1u);
    ret b;
}

fn push_byte(s: &mutable istr, b: u8) {
    s += unsafe_from_byte(b);
}

fn push_bytes(s: &mutable istr, bytes: &[u8]) {
    for byte in bytes {
        push_byte(s, byte);
    }
}

fn split(s: &istr, sep: u8) -> [istr] {
    let v: [istr] = [];
    let accum: istr = ~"";
    let ends_with_sep: bool = false;
    for c: u8 in s {
        if c == sep {
            v += [accum];
            accum = ~"";
            ends_with_sep = true;
        } else { accum += unsafe_from_byte(c); ends_with_sep = false; }
    }
    if byte_len(accum) != 0u || ends_with_sep { v += [accum]; }
    ret v;
}

fn concat(v: &[istr]) -> istr {
    let s: istr = ~"";
    for ss: istr in v { s += ss; }
    ret s;
}

fn connect(v: &[istr], sep: &istr) -> istr {
    let s: istr = ~"";
    let first: bool = true;
    for ss: istr in v {
        if first { first = false; } else { s += sep; }
        s += ss;
    }
    ret s;
}

// FIXME: This only handles ASCII
fn to_upper(s: &istr) -> istr {
    let outstr = ~"";
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
fn replace(s: &istr, from: &istr, to: &istr) : is_not_empty(from) -> istr {
    // FIXME (694): Shouldn't have to check this
    check (is_not_empty(from));
    if byte_len(s) == 0u {
        ret ~"";
    } else if starts_with(s, from) {
        ret to + replace(slice(s, byte_len(from), byte_len(s)), from, to);
    } else {
        ret unsafe_from_byte(s[0]) +
                replace(slice(s, 1u, byte_len(s)), from, to);
    }
}

// FIXME: Also not efficient
fn char_slice(s: &istr, begin: uint, end: uint) -> istr {
    from_chars(vec::slice(to_chars(s), begin, end))
}

fn trim_left(s: &istr) -> istr {
    fn count_whities(s: &[char]) -> uint {
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

fn trim_right(s: &istr) -> istr {
    fn count_whities(s: &[char]) -> uint {
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

fn trim(s: &istr) -> istr {
    trim_left(trim_right(s))
}