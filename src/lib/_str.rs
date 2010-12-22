import rustrt.sbuf;

import std._vec.rustrt.vbuf;

native "rust" mod rustrt {
    type sbuf;
    fn str_buf(str s) -> sbuf;
    fn str_byte_len(str s) -> uint;
    fn str_alloc(uint n_bytes) -> str;
    fn str_from_vec(vec[u8] b) -> str;
    fn refcount[T](str s) -> uint;
}

fn eq(&str a, &str b) -> bool {
    let uint i = byte_len(a);
    if (byte_len(b) != i) {
        ret false;
    }
    while (i > 0u) {
        i -= 1u;
        auto cha = a.(i);
        auto chb = b.(i);
        if (cha != chb) {
            ret false;
        }
    }
    ret true;
}

fn lteq(&str a, &str b) -> bool {
    let uint i = byte_len(a);
    let uint j = byte_len(b);
    let uint n = i;
    if (j < n) {
        n = j;
    }

    let uint x = 0u;
    while (x < n) {
        auto cha = a.(x);
        auto chb = b.(x);
        if (cha <= chb) {
            ret true;
        }
        else if (cha > chb) {
            ret false;
        }
        x += 1u;
    }

    ret i <= j;
}


fn hash(&str s) -> uint {
    // djb hash.
    // FIXME: replace with murmur.
    let uint u = 5381u;
    for (u8 c in s) {
        u *= 33u;
        u += (c as uint);
    }
    ret u;
}

fn is_utf8(vec[u8] v) -> bool {
    fail; // FIXME
}

fn is_ascii(str s) -> bool {
    let uint i = byte_len(s);
    while (i > 0u) {
        i -= 1u;
        if ((s.(i) & 0x80u8) != 0u8) {
            ret false;
        }
    }
    ret true;
}

fn alloc(uint n_bytes) -> str {
    ret rustrt.str_alloc(n_bytes);
}

// Returns the number of bytes (a.k.a. UTF-8 code units) in s.
// Contrast with a function that would return the number of code
// points (char's), combining character sequences, words, etc.  See
// http://icu-project.org/apiref/icu4c/classBreakIterator.html for a
// way to implement those.
fn byte_len(str s) -> uint {
    ret rustrt.str_byte_len(s);
}

fn buf(str s) -> sbuf {
    ret rustrt.str_buf(s);
}

fn bytes(str s) -> vec[u8] {
    /* FIXME (issue #58):
     * Should be...
     *
     *  fn ith(str s, uint i) -> u8 {
     *      ret s.(i);
     *  }
     *  ret _vec.init_fn[u8](bind ith(s, _), byte_len(s));
     *
     * but we do not correctly decrement refcount of s when
     * the binding dies, so we have to do this manually.
     */
    let uint n = _str.byte_len(s);
    let vec[u8] v = _vec.alloc[u8](n);
    let uint i = 0u;
    while (i < n) {
        v += vec(s.(i));
        i += 1u;
    }
    ret v;
}

fn from_bytes(vec[u8] v) : is_utf8(v) -> str {
    ret rustrt.str_from_vec(v);
}

fn refcount(str s) -> uint {
    auto r = rustrt.refcount[u8](s);
    if (r == dbg.const_refcount) {
        ret r;
    } else {
        // -1 because calling this function incremented the refcount.
        ret  r - 1u;
    }
}


// Standard bits from the world of string libraries.

fn index(str s, u8 c) -> int {
    let int i = 0;
    for (u8 k in s) {
        if (k == c) {
            ret i;
        }
        i += 1;
    }
    ret -1;
}

fn rindex(str s, u8 c) -> int {
    let int n = _str.byte_len(s) as int;
    while (n >= 0) {
        if (s.(n) == c) {
            ret n;
        }
        n -= 1;
    }
    ret n;
}

fn find(str haystack, str needle) -> int {

    let int haystack_len = byte_len(haystack) as int;
    let int needle_len = byte_len(needle) as int;

    if (needle_len == 0) {
        ret 0;
    }

    fn match_at(&str haystack,
                &str needle,
                int i) -> bool {
        let int j = i;
        for (u8 c in needle) {
            if (haystack.(j) != c) {
                ret false;
            }
            j += 1;
        }
        ret true;
    }

    let int i = 0;
    while (i <= haystack_len - needle_len) {
        if (match_at(haystack, needle, i)) {
            ret i;
        }
        i += 1;
    }
    ret  -1;
}

fn substr(str s, uint begin, uint len) -> str {
    let str accum = "";
    let uint i = begin;
    while (i < begin+len) {
        accum += s.(i);
        i += 1u;
    }
    ret accum;
}

fn split(str s, u8 sep) -> vec[str] {
    let vec[str] v = vec();
    let str accum = "";
    let bool ends_with_sep = false;
    for (u8 c in s) {
        if (c == sep) {
            v += accum;
            accum = "";
            ends_with_sep = true;
        } else {
            accum += c;
            ends_with_sep = false;
        }
    }
    if (_str.byte_len(accum) != 0u ||
        ends_with_sep) {
        v += accum;
    }
    ret v;
}

fn concat(vec[str] v) -> str {
    let str s = "";
    for (str ss in v) {
        s += ss;
    }
    ret s;
}

fn connect(vec[str] v, str sep) -> str {
    let str s = "";
    let bool first = true;
    for (str ss in v) {
        if (first) {
            first = false;
        } else {
            s += sep;
        }
        s += ss;
    }
    ret s;
}


// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
