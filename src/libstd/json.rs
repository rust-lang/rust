// Rust JSON serialization library
// Copyright (c) 2011 Google Inc.

import float;
import map;
import core::option;
import option::{some, none};
import str;
import vec;

export json;
export to_str;
export from_str;

export num;
export string;
export boolean;
export list;
export dict;

/*
Tag: json

Represents a json value.
*/
tag json {
    /* Variant: num */
    num(float);
    /* Variant: string */
    string(str);
    /* Variant: boolean */
    boolean(bool);
    /* Variant: list */
    list(@[json]);
    /* Variant: dict */
    dict(map::hashmap<str,json>);
}

/*
Function: to_str

Serializes a json value into a string.
*/
fn to_str(j: json) -> str {
    alt j {
        num(f) { float::to_str(f, 6u) }
        string(s) { #fmt["\"%s\"", s] } // XXX: escape
        boolean(true) { "true" }
        boolean(false) { "false" }
        list(@js) {
            str::concat(["[",
                    str::connect(
                        vec::map::<json,str>({ |e| to_str(e) }, js),
                        ", "),
                    "]"])
        }
        dict(m) {
            let parts = [];
            m.items({ |k, v|
                        vec::grow(parts, 1u,
                                  str::concat(["\"", k, "\": ", to_str(v)])
                        )
            });
            str::concat(["{ ", str::connect(parts, ", "), " }"])
        }
    }
}

fn rest(s: str) -> str {
    assert(str::char_len(s) >= 1u);
    str::char_slice(s, 1u, str::char_len(s))
}

fn from_str_str(s: str) -> (option::t<json>, str) {
    let pos = 0u;
    let len = str::byte_len(s);
    let escape = false;
    let res = "";

    alt str::char_at(s, 0u) {
        '"' { pos = 1u; }
        _ { ret (none, s); }
    }

    while (pos < len) {
        let chr = str::char_range_at(s, pos);
        let c = chr.ch;
        pos = chr.next;
        if (escape) {
            res = res + str::from_char(c);
            escape = false;
            cont;
        }
        if (c == '\\') {
            escape = true;
            cont;
        } else if (c == '"') {
            ret (some(string(res)),
                 str::char_slice(s, pos, str::char_len(s)));
        }
        res = res + str::from_char(c);
    }

    ret (none, s);
}

fn from_str_list(s: str) -> (option::t<json>, str) {
    if str::char_at(s, 0u) != '[' { ret (none, s); }
    let s0 = str::trim_left(rest(s));
    let vals = [];
    if str::is_empty(s0) { ret (none, s0); }
    if str::char_at(s0, 0u) == ']' { ret (some(list(@[])), rest(s0)); }
    while str::is_not_empty(s0) {
        s0 = str::trim_left(s0);
        let (next, s1) = from_str_helper(s0);
        s0 = s1;
        alt next {
            some(j) { vec::grow(vals, 1u, j); }
            none { ret (none, s0); }
        }
        s0 = str::trim_left(s0);
        if str::is_empty(s0) { ret (none, s0); }
        alt str::char_at(s0, 0u) {
            ',' { }
            ']' { ret (some(list(@vals)), rest(s0)); }
            _ { ret (none, s0); }
        }
        s0 = rest(s0);
    }
    ret (none, s0);
}

fn from_str_dict(s: str) -> (option::t<json>, str) {
    if str::char_at(s, 0u) != '{' { ret (none, s); }
    let s0 = str::trim_left(rest(s));
    let vals = map::new_str_hash::<json>();
    if str::is_empty(s0) { ret (none, s0); }
    if str::char_at(s0, 0u) == '}' { ret (some(dict(vals)), rest(s0)); }
    while str::is_not_empty(s0) {
        s0 = str::trim_left(s0);
        let (next, s1) = from_str_helper(s0);    // key
        let key = "";
        s0 = s1;
        alt next {
            some(string(k)) { key = k; }
            _ { ret (none, s0); }
        }
        s0 = str::trim_left(s0);
        if str::is_empty(s0) { ret (none, s0); }
        if str::char_at(s0, 0u) != ':' { ret (none, s0); }
        s0 = str::trim_left(rest(s0));
        let (next, s1) = from_str_helper(s0);    // value
        s0 = s1;
        alt next {
            some(j) { vals.insert(key, j); }
            _ { ret (none, s0); }
        }
        s0 = str::trim_left(s0);
        if str::is_empty(s0) { ret (none, s0); }
        alt str::char_at(s0, 0u) {
            ',' { }
            '}' { ret (some(dict(vals)), rest(s0)); }
            _ { ret (none, s0); }
        }
        s0 = str::trim_left(rest(s0));
    }
    (none, s)
}

fn from_str_float(s: str) -> (option::t<json>, str) {
    let pos = 0u;
    let len = str::byte_len(s);
    let res = 0f;
    let neg = 1.f;

    alt str::char_at(s, 0u) {
        '-' {
            neg = -1.f;
            pos = 1u;
        }
        '+' {
            pos = 1u;
        }
        '0' to '9' | '.' { }
        _ { ret (none, s); }
    }

    while (pos < len) {
        let opos = pos;
        let chr = str::char_range_at(s, pos);
        let c = chr.ch;
        pos = chr.next;
        alt c {
            '0' to '9' {
                res = res * 10f;
                res += ((c as int) - ('0' as int)) as float;
            }
            '.' { break; }
            _ { ret (some(num(neg * res)),
                     str::char_slice(s, opos, str::char_len(s))); }
        }
    }

    if pos == len {
        ret (some(num(neg * res)), str::char_slice(s, pos, str::char_len(s)));
    }

    let dec = 1.f;
    while (pos < len) {
        let opos = pos;
        let chr = str::char_range_at(s, pos);
        let c = chr.ch;
        pos = chr.next;
        alt c {
            '0' to '9' {
                dec /= 10.f;
                res += (((c as int) - ('0' as int)) as float) * dec;
            }
            _ { ret (some(num(neg * res)),
                     str::char_slice(s, opos, str::char_len(s))); }
        }
    }
    ret (some(num(neg * res)), str::char_slice(s, pos, str::char_len(s)));
}

fn from_str_bool(s: str) -> (option::t<json>, str) {
    if (str::starts_with(s, "true")) {
        (some(boolean(true)), str::slice(s, 4u, str::byte_len(s)))
    } else if (str::starts_with(s, "false")) {
        (some(boolean(false)), str::slice(s, 5u, str::byte_len(s)))
    } else {
        (none, s)
    }
}

fn from_str_helper(s: str) -> (option::t<json>, str) {
    let s = str::trim_left(s);
    if str::is_empty(s) { ret (none, s); }
    let start = str::char_at(s, 0u);
    alt start {
        '"' { from_str_str(s) }
        '[' { from_str_list(s) }
        '{' { from_str_dict(s) }
        '0' to '9' | '-' | '+' | '.' { from_str_float(s) }
        't' | 'f' { from_str_bool(s) }
        _ { ret (none, s); }
    }
}

/*
Function: from_str

Deserializes a json value from a string.
*/
fn from_str(s: str) -> option::t<json> {
    let (j, _) = from_str_helper(s);
    j
}
