
import rustrt::sbuf;
import vec::rustrt::vbuf;
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
export unsafe_from_bytes;
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
export concat;
export connect;
export to_upper;

native "rust" mod rustrt {
    type sbuf;
    fn str_buf(str s) -> sbuf;
    fn str_vec(str s) -> vec[u8];
    fn str_byte_len(str s) -> uint;
    fn str_alloc(uint n_bytes) -> str;
    fn str_from_vec(vec[mutable? u8] b) -> str;
    fn str_from_cstr(sbuf cstr) -> str;
    fn str_from_buf(sbuf buf, uint len) -> str;
    fn str_push_byte(str s, uint byte) -> str;
    fn str_slice(str s, uint begin, uint end) -> str;
    fn refcount[T](str s) -> uint;
}

fn eq(&str a, &str b) -> bool {
    let uint i = byte_len(a);
    if (byte_len(b) != i) { ret false; }
    while (i > 0u) {
        i -= 1u;
        auto cha = a.(i);
        auto chb = b.(i);
        if (cha != chb) { ret false; }
    }
    ret true;
}

fn lteq(&str a, &str b) -> bool {
    let uint i = byte_len(a);
    let uint j = byte_len(b);
    let uint n = i;
    if (j < n) { n = j; }
    let uint x = 0u;
    while (x < n) {
        auto cha = a.(x);
        auto chb = b.(x);
        if (cha < chb) { ret true; } else if (cha > chb) { ret false; }
        x += 1u;
    }
    ret i <= j;
}

fn hash(&str s) -> uint {
    // djb hash.
    // FIXME: replace with murmur.

    let uint u = 5381u;
    for (u8 c in s) { u *= 33u; u += c as uint; }
    ret u;
}


// UTF-8 tags and ranges
const u8 tag_cont_u8 = 128u8;

const uint tag_cont = 128u;

const uint max_one_b = 128u;

const uint tag_two_b = 192u;

const uint max_two_b = 2048u;

const uint tag_three_b = 224u;

const uint max_three_b = 65536u;

const uint tag_four_b = 240u;

const uint max_four_b = 2097152u;

const uint tag_five_b = 248u;

const uint max_five_b = 67108864u;

const uint tag_six_b = 252u;

fn is_utf8(vec[u8] v) -> bool {
    auto i = 0u;
    auto total = vec::len[u8](v);
    while (i < total) {
        auto chsize = utf8_char_width(v.(i));
        if (chsize == 0u) { ret false; }
        if (i + chsize > total) { ret false; }
        i += 1u;
        while (chsize > 1u) {
            if (v.(i) & 192u8 != tag_cont_u8) { ret false; }
            i += 1u;
            chsize -= 1u;
        }
    }
    ret true;
}

fn is_ascii(str s) -> bool {
    let uint i = byte_len(s);
    while (i > 0u) { i -= 1u; if (s.(i) & 128u8 != 0u8) { ret false; } }
    ret true;
}

fn alloc(uint n_bytes) -> str { ret rustrt::str_alloc(n_bytes); }


// Returns the number of bytes (a.k.a. UTF-8 code units) in s.
// Contrast with a function that would return the number of code
// points (char's), combining character sequences, words, etc.  See
// http://icu-project.org/apiref/icu4c/classBreakIterator.html for a
// way to implement those.
fn byte_len(str s) -> uint { ret rustrt::str_byte_len(s); }

fn buf(str s) -> sbuf { ret rustrt::str_buf(s); }

fn bytes(str s) -> vec[u8] { ret rustrt::str_vec(s); }

fn from_bytes(vec[u8] v) -> str { ret rustrt::str_from_vec(v); }


// FIXME temp thing
fn unsafe_from_bytes(vec[mutable? u8] v) -> str {
    ret rustrt::str_from_vec(v);
}

fn unsafe_from_byte(u8 u) -> str { ret rustrt::str_from_vec([u]); }

fn str_from_cstr(sbuf cstr) -> str { ret rustrt::str_from_cstr(cstr); }

fn str_from_buf(sbuf buf, uint len) -> str {
    ret rustrt::str_from_buf(buf, len);
}

fn push_utf8_bytes(&mutable str s, char ch) {
    auto code = ch as uint;
    if (code < max_one_b) {
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

fn from_char(char ch) -> str {
    auto buf = "";
    push_utf8_bytes(buf, ch);
    ret buf;
}

fn from_chars(vec[char] chs) -> str {
    auto buf = "";
    for (char ch in chs) { push_utf8_bytes(buf, ch); }
    ret buf;
}

fn utf8_char_width(u8 b) -> uint {
    let uint byte = b as uint;
    if (byte < 128u) { ret 1u; }
    if (byte < 192u) {
        ret 0u; // Not a valid start byte

    }
    if (byte < 224u) { ret 2u; }
    if (byte < 240u) { ret 3u; }
    if (byte < 248u) { ret 4u; }
    if (byte < 252u) { ret 5u; }
    ret 6u;
}

fn char_range_at(str s, uint i) -> tup(char, uint) {
    auto b0 = s.(i);
    auto w = utf8_char_width(b0);
    assert (w != 0u);
    if (w == 1u) { ret tup(b0 as char, i + 1u); }
    auto val = 0u;
    auto end = i + w;
    i += 1u;
    while (i < end) {
        auto byte = s.(i);
        assert (byte & 192u8 == tag_cont_u8);
        val <<= 6u;
        val += byte & 63u8 as uint;
        i += 1u;
    }
    // Clunky way to get the right bits from the first byte. Uses two shifts,
    // the first to clip off the marker bits at the left of the byte, and then
    // a second (as uint) to get it to the right position.

    val += (b0 << (w + 1u as u8) as uint) << (w - 1u) * 6u - w - 1u;
    ret tup(val as char, i);
}

fn char_at(str s, uint i) -> char { ret char_range_at(s, i)._0; }

fn char_len(str s) -> uint {
    auto i = 0u;
    auto len = 0u;
    auto total = byte_len(s);
    while (i < total) {
        auto chsize = utf8_char_width(s.(i));
        assert (chsize > 0u);
        len += 1u;
        i += chsize;
    }
    assert (i == total);
    ret len;
}

fn to_chars(str s) -> vec[char] {
    let vec[char] buf = [];
    auto i = 0u;
    auto len = byte_len(s);
    while (i < len) {
        auto cur = char_range_at(s, i);
        vec::push[char](buf, cur._0);
        i = cur._1;
    }
    ret buf;
}

fn push_char(&mutable str s, char ch) { s += from_char(ch); }

fn pop_char(&mutable str s) -> char {
    auto end = byte_len(s);
    while (end > 0u && s.(end - 1u) & 192u8 == tag_cont_u8) { end -= 1u; }
    assert (end > 0u);
    auto ch = char_at(s, end - 1u);
    s = substr(s, 0u, end - 1u);
    ret ch;
}

fn shift_char(&mutable str s) -> char {
    auto r = char_range_at(s, 0u);
    s = substr(s, r._1, byte_len(s) - r._1);
    ret r._0;
}

fn unshift_char(&mutable str s, char ch) {
    // Workaround for rustboot order-of-evaluation issue -- if I put s
    // directly after the +, the string ends up containing (only) the
    // character, twice.

    auto x = s;
    s = from_char(ch) + x;
}

fn refcount(str s) -> uint {
    auto r = rustrt::refcount[u8](s);
    if (r == dbg::const_refcount) {
        ret r;
    } else {
        // -2 because calling this function and the native function both
        // incremented the refcount.

        ret r - 2u;
    }
}


// Standard bits from the world of string libraries.
fn index(str s, u8 c) -> int {
    let int i = 0;
    for (u8 k in s) { if (k == c) { ret i; } i += 1; }
    ret -1;
}

fn rindex(str s, u8 c) -> int {
    let int n = str::byte_len(s) as int;
    while (n >= 0) { if (s.(n) == c) { ret n; } n -= 1; }
    ret n;
}

fn find(str haystack, str needle) -> int {
    let int haystack_len = byte_len(haystack) as int;
    let int needle_len = byte_len(needle) as int;
    if (needle_len == 0) { ret 0; }
    fn match_at(&str haystack, &str needle, int i) -> bool {
        let int j = i;
        for (u8 c in needle) { if (haystack.(j) != c) { ret false; } j += 1; }
        ret true;
    }
    let int i = 0;
    while (i <= haystack_len - needle_len) {
        if (match_at(haystack, needle, i)) { ret i; }
        i += 1;
    }
    ret -1;
}

fn starts_with(str haystack, str needle) -> bool {
    let uint haystack_len = byte_len(haystack);
    let uint needle_len = byte_len(needle);
    if (needle_len == 0u) { ret true; }
    if (needle_len > haystack_len) { ret false; }
    ret eq(substr(haystack, 0u, needle_len), needle);
}

fn ends_with(str haystack, str needle) -> bool {
    let uint haystack_len = byte_len(haystack);
    let uint needle_len = byte_len(needle);
    ret if (needle_len == 0u) {
            true
        } else if (needle_len > haystack_len) {
            false
        } else {
            eq(substr(haystack, haystack_len - needle_len, needle_len),
               needle)
        };
}

fn substr(str s, uint begin, uint len) -> str {
    ret slice(s, begin, begin + len);
}

fn slice(str s, uint begin, uint end) -> str {
    // FIXME: Typestate precondition

    assert (begin <= end);
    assert (end <= str::byte_len(s));
    ret rustrt::str_slice(s, begin, end);
}

fn shift_byte(&mutable str s) -> u8 {
    auto len = byte_len(s);
    assert (len > 0u);
    auto b = s.(0);
    s = substr(s, 1u, len - 1u);
    ret b;
}

fn pop_byte(&mutable str s) -> u8 {
    auto len = byte_len(s);
    assert (len > 0u);
    auto b = s.(len - 1u);
    s = substr(s, 0u, len - 1u);
    ret b;
}

fn push_byte(&mutable str s, u8 b) {
    s = rustrt::str_push_byte(s, b as uint);
}

fn unshift_byte(&mutable str s, u8 b) {
    auto res = alloc(byte_len(s) + 1u);
    res += unsafe_from_byte(b);
    res += s;
    s = res;
}

fn split(str s, u8 sep) -> vec[str] {
    let vec[str] v = [];
    let str accum = "";
    let bool ends_with_sep = false;
    for (u8 c in s) {
        if (c == sep) {
            v += [accum];
            accum = "";
            ends_with_sep = true;
        } else { accum += unsafe_from_byte(c); ends_with_sep = false; }
    }
    if (str::byte_len(accum) != 0u || ends_with_sep) { v += [accum]; }
    ret v;
}

fn concat(vec[str] v) -> str {
    let str s = "";
    for (str ss in v) { s += ss; }
    ret s;
}

fn connect(vec[str] v, str sep) -> str {
    let str s = "";
    let bool first = true;
    for (str ss in v) {
        if (first) { first = false; } else { s += sep; }
        s += ss;
    }
    ret s;
}


// FIXME: This only handles ASCII
fn to_upper(str s) -> str {
    auto outstr = "";
    auto ascii_a = 'a' as u8;
    auto ascii_z = 'z' as u8;
    auto diff = 32u8;
    for (u8 byte in s) {
        auto next;
        if (ascii_a <= byte && byte <= ascii_z) {
            next = byte - diff;
        } else { next = byte; }
        push_byte(outstr, next);
    }
    ret outstr;
}
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
