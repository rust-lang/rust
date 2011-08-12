
import rustrt::sbuf;
import vec::rustrt::vbuf;
import uint::le;
export sbuf;
export rustrt;
export eq;
export lteq;
export hash;
export is_utf8;
export is_ascii;
export alloc;
export byte_len;
export buf;
export bytes;
export from_bytes;
export unsafe_from_byte;
export str_from_cstr;
export str_from_buf;
export push_utf8_bytes;
export from_char;
export from_chars;
export utf8_char_width;
export char_range_at;
export char_at;
export char_len;
export to_chars;
export push_char;
export pop_char;
export shift_char;
export unshift_char;
export refcount;
export index;
export rindex;
export find;
export starts_with;
export ends_with;
export substr;
export slice;
export shift_byte;
export pop_byte;
export push_byte;
export unshift_byte;
export split;
export split_ivec;
export concat;
export connect;
export connect_ivec;
export to_upper;
export safe_slice;
export unsafe_from_bytes_ivec;
export is_empty;
export is_not_empty;
export is_whitespace;
export replace;
export char_slice;
export trim_left;
export trim_right;
export trim;

native "rust" mod rustrt {
    type sbuf;
    fn str_buf(s: str) -> sbuf;
    fn str_byte_len(s: str) -> uint;
    fn str_alloc(n_bytes: uint) -> str;
    fn str_from_ivec(b: &[mutable? u8]) -> str;
    fn str_from_vec(b: vec[mutable? u8]) -> str;
    fn str_from_cstr(cstr: sbuf) -> str;
    fn str_from_buf(buf: sbuf, len: uint) -> str;
    fn str_push_byte(s: str, byte: uint) -> str;
    fn str_slice(s: str, begin: uint, end: uint) -> str;
    fn refcount[T](s: str) -> uint;
}

fn eq(a: &str, b: &str) -> bool {
    let i: uint = byte_len(a);
    if byte_len(b) != i { ret false; }
    while i > 0u {
        i -= 1u;
        let cha = a.(i);
        let chb = b.(i);
        if cha != chb { ret false; }
    }
    ret true;
}

fn lteq(a: &str, b: &str) -> bool {
    let i: uint = byte_len(a);
    let j: uint = byte_len(b);
    let n: uint = i;
    if j < n { n = j; }
    let x: uint = 0u;
    while x < n {
        let cha = a.(x);
        let chb = b.(x);
        if cha < chb { ret true; } else if (cha > chb) { ret false; }
        x += 1u;
    }
    ret i <= j;
}

fn hash(s: &str) -> uint {
    // djb hash.
    // FIXME: replace with murmur.

    let u: uint = 5381u;
    for c: u8  in s { u *= 33u; u += c as uint; }
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
    let total = ivec::len[u8](v);
    while i < total {
        let chsize = utf8_char_width(v.(i));
        if chsize == 0u { ret false; }
        if i + chsize > total { ret false; }
        i += 1u;
        while chsize > 1u {
            if v.(i) & 192u8 != tag_cont_u8 { ret false; }
            i += 1u;
            chsize -= 1u;
        }
    }
    ret true;
}

fn is_ascii(s: str) -> bool {
    let i: uint = byte_len(s);
    while i > 0u { i -= 1u; if s.(i) & 128u8 != 0u8 { ret false; } }
    ret true;
}

fn alloc(n_bytes: uint) -> str { ret rustrt::str_alloc(n_bytes); }

/// Returns true if the string has length 0
pred is_empty(s: str) -> bool { for c: u8  in s { ret false; } ret true; }

/// Returns true if the string has length greater than 0
pred is_not_empty(s: str) -> bool { !is_empty(s) }

fn is_whitespace(s: str) -> bool {
    let i = 0u;
    let len = char_len(s);
    while i < len {
        if !char::is_whitespace(char_at(s, i)) {
            ret false;
        }
        i += 1u
    }
    ret true;
}

// Returns the number of bytes (a.k.a. UTF-8 code units) in s.
// Contrast with a function that would return the number of code
// points (char's), combining character sequences, words, etc.  See
// http://icu-project.org/apiref/icu4c/classBreakIterator.html for a
// way to implement those.
fn byte_len(s: str) -> uint { ret rustrt::str_byte_len(s); }

fn buf(s: &str) -> sbuf { ret rustrt::str_buf(s); }

fn bytes(s: str) -> [u8] {
    let sbuffer = buf(s);
    let ptr = unsafe::reinterpret_cast(sbuffer);
    ret ivec::unsafe::from_buf(ptr, byte_len(s));
}

fn unsafe_from_bytes_ivec(v: &[mutable? u8]) -> str {
    ret rustrt::str_from_ivec(v);
}

fn unsafe_from_byte(u: u8) -> str { ret rustrt::str_from_vec([u]); }

fn str_from_cstr(cstr: sbuf) -> str { ret rustrt::str_from_cstr(cstr); }

fn str_from_buf(buf: sbuf, len: uint) -> str {
    ret rustrt::str_from_buf(buf, len);
}

fn push_utf8_bytes(s: &mutable str, ch: char) {
    let code = ch as uint;
    if code < max_one_b {
        s = rustrt::str_push_byte(s, code);
    } else if (code < max_two_b) {
        s = rustrt::str_push_byte(s, code >> 6u & 31u | tag_two_b);
        s = rustrt::str_push_byte(s, code & 63u | tag_cont);
    } else if (code < max_three_b) {
        s = rustrt::str_push_byte(s, code >> 12u & 15u | tag_three_b);
        s = rustrt::str_push_byte(s, code >> 6u & 63u | tag_cont);
        s = rustrt::str_push_byte(s, code & 63u | tag_cont);
    } else if (code < max_four_b) {
        s = rustrt::str_push_byte(s, code >> 18u & 7u | tag_four_b);
        s = rustrt::str_push_byte(s, code >> 12u & 63u | tag_cont);
        s = rustrt::str_push_byte(s, code >> 6u & 63u | tag_cont);
        s = rustrt::str_push_byte(s, code & 63u | tag_cont);
    } else if (code < max_five_b) {
        s = rustrt::str_push_byte(s, code >> 24u & 3u | tag_five_b);
        s = rustrt::str_push_byte(s, code >> 18u & 63u | tag_cont);
        s = rustrt::str_push_byte(s, code >> 12u & 63u | tag_cont);
        s = rustrt::str_push_byte(s, code >> 6u & 63u | tag_cont);
        s = rustrt::str_push_byte(s, code & 63u | tag_cont);
    } else {
        s = rustrt::str_push_byte(s, code >> 30u & 1u | tag_six_b);
        s = rustrt::str_push_byte(s, code >> 24u & 63u | tag_cont);
        s = rustrt::str_push_byte(s, code >> 18u & 63u | tag_cont);
        s = rustrt::str_push_byte(s, code >> 12u & 63u | tag_cont);
        s = rustrt::str_push_byte(s, code >> 6u & 63u | tag_cont);
        s = rustrt::str_push_byte(s, code & 63u | tag_cont);
    }
}

fn from_char(ch: char) -> str {
    let buf = "";
    push_utf8_bytes(buf, ch);
    ret buf;
}

fn from_chars(chs: vec[char]) -> str {
    let buf = "";
    for ch: char  in chs { push_utf8_bytes(buf, ch); }
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

fn char_range_at(s: str, i: uint) -> {ch: char, next: uint} {
    let b0 = s.(i);
    let w = utf8_char_width(b0);
    assert (w != 0u);
    if w == 1u { ret {ch: b0 as char, next: i + 1u}; }
    let val = 0u;
    let end = i + w;
    i += 1u;
    while i < end {
        let byte = s.(i);
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

fn char_at(s: str, i: uint) -> char { ret char_range_at(s, i).ch; }

fn char_len(s: str) -> uint {
    let i = 0u;
    let len = 0u;
    let total = byte_len(s);
    while i < total {
        let chsize = utf8_char_width(s.(i));
        assert (chsize > 0u);
        len += 1u;
        i += chsize;
    }
    assert (i == total);
    ret len;
}

fn to_chars(s: str) -> vec[char] {
    let buf: vec[char] = [];
    let i = 0u;
    let len = byte_len(s);
    while i < len {
        let cur = char_range_at(s, i);
        vec::push[char](buf, cur.ch);
        i = cur.next;
    }
    ret buf;
}

fn push_char(s: &mutable str, ch: char) { s += from_char(ch); }

fn pop_char(s: &mutable str) -> char {
    let end = byte_len(s);
    while end > 0u && s.(end - 1u) & 192u8 == tag_cont_u8 { end -= 1u; }
    assert (end > 0u);
    let ch = char_at(s, end - 1u);
    s = substr(s, 0u, end - 1u);
    ret ch;
}

fn shift_char(s: &mutable str) -> char {
    let r = char_range_at(s, 0u);
    s = substr(s, r.next, byte_len(s) - r.next);
    ret r.ch;
}

fn unshift_char(s: &mutable str, ch: char) { s = from_char(ch) + s; }

fn refcount(s: str) -> uint {
    let r = rustrt::refcount[u8](s);
    if r == dbg::const_refcount { ret r; } else { ret r - 1u; }
}


// Standard bits from the world of string libraries.
fn index(s: str, c: u8) -> int {
    let i: int = 0;
    for k: u8  in s { if k == c { ret i; } i += 1; }
    ret -1;
}

fn rindex(s: str, c: u8) -> int {
    let n: int = str::byte_len(s) as int;
    while n >= 0 { if s.(n) == c { ret n; } n -= 1; }
    ret n;
}

fn find(haystack: str, needle: str) -> int {
    let haystack_len: int = byte_len(haystack) as int;
    let needle_len: int = byte_len(needle) as int;
    if needle_len == 0 { ret 0; }
    fn match_at(haystack: &str, needle: &str, i: int) -> bool {
        let j: int = i;
        for c: u8  in needle { if haystack.(j) != c { ret false; } j += 1; }
        ret true;
    }
    let i: int = 0;
    while i <= haystack_len - needle_len {
        if match_at(haystack, needle, i) { ret i; }
        i += 1;
    }
    ret -1;
}

fn starts_with(haystack: str, needle: str) -> bool {
    let haystack_len: uint = byte_len(haystack);
    let needle_len: uint = byte_len(needle);
    if needle_len == 0u { ret true; }
    if needle_len > haystack_len { ret false; }
    ret eq(substr(haystack, 0u, needle_len), needle);
}

fn ends_with(haystack: str, needle: str) -> bool {
    let haystack_len: uint = byte_len(haystack);
    let needle_len: uint = byte_len(needle);
    ret if needle_len == 0u {
            true
        } else if (needle_len > haystack_len) {
            false
        } else {
            eq(substr(haystack, haystack_len - needle_len, needle_len),
               needle)
        };
}

fn substr(s: str, begin: uint, len: uint) -> str {
    ret slice(s, begin, begin + len);
}

fn slice(s: str, begin: uint, end: uint) -> str {
    // FIXME: Typestate precondition

    assert (begin <= end);
    assert (end <= str::byte_len(s));
    ret rustrt::str_slice(s, begin, end);
}

fn safe_slice(s: str, begin: uint, end: uint): le(begin, end) -> str {
    assert (end <=
                str::byte_len(s)); // would need some magic to
                                   // make this a precondition

    ret rustrt::str_slice(s, begin, end);
}

fn shift_byte(s: &mutable str) -> u8 {
    let len = byte_len(s);
    assert (len > 0u);
    let b = s.(0);
    s = substr(s, 1u, len - 1u);
    ret b;
}

fn pop_byte(s: &mutable str) -> u8 {
    let len = byte_len(s);
    assert (len > 0u);
    let b = s.(len - 1u);
    s = substr(s, 0u, len - 1u);
    ret b;
}

fn push_byte(s: &mutable str, b: u8) {
    s = rustrt::str_push_byte(s, b as uint);
}

fn unshift_byte(s: &mutable str, b: u8) {
    let rs = alloc(byte_len(s) + 1u);
    rs += unsafe_from_byte(b);
    rs += s;
    s = rs;
}

fn split(s: str, sep: u8) -> vec[str] {
    let v: vec[str] = [];
    let accum: str = "";
    let ends_with_sep: bool = false;
    for c: u8  in s {
        if c == sep {
            v += [accum];
            accum = "";
            ends_with_sep = true;
        } else { accum += unsafe_from_byte(c); ends_with_sep = false; }
    }
    if str::byte_len(accum) != 0u || ends_with_sep { v += [accum]; }
    ret v;
}

fn split_ivec(s: str, sep: u8) -> [str] {
    let v: [str] = ~[];
    let accum: str = "";
    let ends_with_sep: bool = false;
    for c: u8  in s {
        if c == sep {
            v += ~[accum];
            accum = "";
            ends_with_sep = true;
        } else { accum += unsafe_from_byte(c); ends_with_sep = false; }
    }
    if str::byte_len(accum) != 0u || ends_with_sep { v += ~[accum]; }
    ret v;
}

fn concat(v: vec[str]) -> str {
    let s: str = "";
    for ss: str  in v { s += ss; }
    ret s;
}

fn connect(v: vec[str], sep: str) -> str {
    let s: str = "";
    let first: bool = true;
    for ss: str  in v {
        if first { first = false; } else { s += sep; }
        s += ss;
    }
    ret s;
}

fn connect_ivec(v: &[str], sep: str) -> str {
    let s: str = "";
    let first: bool = true;
    for ss: str  in v {
        if first { first = false; } else { s += sep; }
        s += ss;
    }
    ret s;
}


// FIXME: This only handles ASCII
fn to_upper(s: str) -> str {
    let outstr = "";
    let ascii_a = 'a' as u8;
    let ascii_z = 'z' as u8;
    let diff = 32u8;
    for byte: u8  in s {
        let next;
        if ascii_a <= byte && byte <= ascii_z {
            next = byte - diff;
        } else { next = byte; }
        push_byte(outstr, next);
    }
    ret outstr;
}

// FIXME: This is super-inefficient
fn replace(s: str, from: str, to: str) : is_not_empty(from) -> str {
    // FIXME (694): Shouldn't have to check this
    check (is_not_empty(from));
    if byte_len(s) == 0u {
        ret "";
    } else if (starts_with(s, from)) {
        ret to + replace(slice(s, byte_len(from), byte_len(s)), from, to);
    } else {
        ret unsafe_from_byte(s.(0)) +
                replace(slice(s, 1u, byte_len(s)), from, to);
    }
}

// FIXME: Also not efficient
fn char_slice(s: &str, begin: uint, end: uint) -> str {
    from_chars(vec::slice(to_chars(s), begin, end))
}

fn trim_left(s: &str) -> str {
    fn count_whities(s: &vec[char]) -> uint {
        let i = 0u;
        while i < vec::len(s) {
            if !char::is_whitespace(s.(i)) {
                break;
            }
            i += 1u;
        }
        ret i;
    }
    let chars = to_chars(s);
    let whities = count_whities(chars);
    ret from_chars(vec::slice(chars, whities, vec::len(chars)));
}

fn trim_right(s: &str) -> str {
    fn count_whities(s: &vec[char]) -> uint {
        let i = vec::len(s);
        while 0u < i {
            if !char::is_whitespace(s.(i - 1u)) {
                break;
            }
            i -= 1u;
        }
        ret i;
    }
    let chars = to_chars(s);
    let whities = count_whities(chars);
    ret from_chars(vec::slice(chars, 0u, whities));
}

fn trim(s: &str) -> str {
    trim_left(trim_right(s))
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
