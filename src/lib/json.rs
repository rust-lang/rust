#[link(name = "json",
       vers = "0.1",
       uuid = "09d7f2fc-1fad-48b2-9f9d-a65512342f16",
       url = "http://www.leptoquark.net/~elly/rust/")];
#[copyright = "Google Inc. 2011"];
#[comment = "JSON serialization format library"];
#[license = "BSD"];

import float;
import map;
import option;
import option::{some, none};
import str;
import vec;

export json;
export tostr;
export fromstr;

tag json {
    num(float);
    string(str);
    boolean(bool);
    list(@[json]);
    dict(map::hashmap<str,json>);
}

fn tostr(j: json) -> str {
    alt j {
        num(f) { float::to_str(f, 6u) }
        string(s) { #fmt["\"%s\"", s] } // XXX: escape
        boolean(true) { "true" }
        boolean(false) { "false" }
        list(@js) {
            str::concat(["[",
                    str::connect(
                        vec::map::<json,str>({ |e| tostr(e) }, js),
                        ", "),
                    "]"])
        }
        dict(m) {
            let parts = [];
            m.items({ |k, v|
                        vec::grow(parts, 1u,
                                  str::concat(["\"", k, "\": ", tostr(v)])
                        )
            });
            str::concat(["{ ", str::connect(parts, ", "), " }"])
        }
    }
}

fn rest(s: str) -> str {
    str::char_slice(s, 1u, str::char_len(s))
}

fn fromstr_str(s: str) -> (option::t<json>, str) {
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
            ret (some(string(res)), str::char_slice(s, pos, str::char_len(s)));
        }
        res = res + str::from_char(c);
    }

    ret (none, s);
}

fn fromstr_list(s: str) -> (option::t<json>, str) {
    if str::char_at(s, 0u) != '[' { ret (none, s); }
    let s0 = str::trim_left(rest(s));
    let vals = [];
    if str::is_empty(s0) { ret (none, s0); }
    if str::char_at(s0, 0u) == ']' { ret (some(list(@[])), rest(s0)); }
    while str::is_not_empty(s0) {
        s0 = str::trim_left(s0);
        let (next, s1) = fromstr_helper(s0);
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

fn fromstr_dict(s: str) -> (option::t<json>, str) {
    if str::char_at(s, 0u) != '{' { ret (none, s); }
    let s0 = str::trim_left(rest(s));
    let vals = map::new_str_hash::<json>();
    if str::is_empty(s0) { ret (none, s0); }
    if str::char_at(s0, 0u) == '}' { ret (some(dict(vals)), rest(s0)); }
    while str::is_not_empty(s0) {
        s0 = str::trim_left(s0);
        let (next, s1) = fromstr_helper(s0);    // key
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
        let (next, s1) = fromstr_helper(s0);    // value
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

fn fromstr_float(s: str) -> (option::t<json>, str) {
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

fn fromstr_bool(s: str) -> (option::t<json>, str) {
    if (str::starts_with(s, "true")) {
        (some(boolean(true)), str::slice(s, 4u, str::byte_len(s)))
    } else if (str::starts_with(s, "false")) {
        (some(boolean(false)), str::slice(s, 5u, str::byte_len(s)))
    } else {
        (none, s)
    }
}

fn fromstr_helper(s: str) -> (option::t<json>, str) {
    let s = str::trim_left(s);
    if str::is_empty(s) { ret (none, s); }
    let start = str::char_at(s, 0u);
    alt start {
        '"' { fromstr_str(s) }
        '[' { fromstr_list(s) }
        '{' { fromstr_dict(s) }
        '0' to '9' | '-' | '+' | '.' { fromstr_float(s) }
        't' | 'f' { fromstr_bool(s) }
        _ { ret (none, s); }
    }
}

fn fromstr(s: str) -> option::t<json> {
    let (j, _) = fromstr_helper(s);
    j
}

fn main() {
    let j = fromstr("{ \"foo\": [ 4, 5 ], \"bar\": { \"baz\": true}}");
    alt j {
        some(j0) {
            log tostr(j0);
        }
        _ { }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_fromstr_num() {
        assert(fromstr("3") == some(num(3f)));
        assert(fromstr("3.1") == some(num(3.1f)));
        assert(fromstr("-1.2") == some(num(-1.2f)));
        assert(fromstr(".4") == some(num(0.4f)));
    }

    #[test]
    fn test_fromstr_str() {
        assert(fromstr("\"foo\"") == some(string("foo")));
        assert(fromstr("\"\\\"\"") == some(string("\"")));
        assert(fromstr("\"lol") == none);
    }

    #[test]
    fn test_fromstr_bool() {
        assert(fromstr("true") == some(boolean(true)));
        assert(fromstr("false") == some(boolean(false)));
        assert(fromstr("truz") == none);
    }

    #[test]
    fn test_fromstr_list() {
        assert(fromstr("[]") == some(list(@[])));
        assert(fromstr("[true]") == some(list(@[boolean(true)])));
        assert(fromstr("[3, 1]") == some(list(@[num(3f), num(1f)])));
        assert(fromstr("[2, [4, 1]]") ==
            some(list(@[num(2f), list(@[num(4f), num(1f)])])));
        assert(fromstr("[2, ]") == none);
        assert(fromstr("[5, ") == none);
        assert(fromstr("[6 7]") == none);
        assert(fromstr("[3") == none);
    }

    #[test]
    fn test_fromstr_dict() {
        assert(fromstr("{}") != none);
        assert(fromstr("{\"a\": 3}") != none);
        assert(fromstr("{\"a\": }") == none);
        assert(fromstr("{\"a\" }") == none);
        assert(fromstr("{\"a\"") == none);
        assert(fromstr("{") == none);
    }
}
