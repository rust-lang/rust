//! Types/fns concerning URLs (see RFC 3986)

export url, userinfo, query, from_str, to_str, get_scheme;

type url = {
    scheme: ~str,
    user: option<userinfo>,
    host: ~str,
    path: ~str,
    query: query,
    fragment: option<~str>
};

type userinfo = {
    user: ~str,
    pass: option<~str>
};

type query = ~[(~str, ~str)];

fn url(-scheme: ~str, -user: option<userinfo>, -host: ~str,
       -path: ~str, -query: query, -fragment: option<~str>) -> url {
    { scheme: scheme, user: user, host: host,
     path: path, query: query, fragment: fragment }
}

fn userinfo(-user: ~str, -pass: option<~str>) -> userinfo {
    {user: user, pass: pass}
}

fn split_char_first(s: ~str, c: char) -> (~str, ~str) {
    let mut v = str::splitn_char(s, c, 1);
    if v.len() == 1 {
        ret (s, ~"");
    } else {
        ret (vec::shift(v), vec::pop(v));
    }
}

fn userinfo_from_str(uinfo: ~str) -> userinfo {
    let (user, p) = split_char_first(uinfo, ':');
    let pass = if str::len(p) == 0 {
        option::none
    } else {
        option::some(p)
    };
    ret userinfo(user, pass);
}

fn userinfo_to_str(-userinfo: userinfo) -> ~str {
    if option::is_some(userinfo.pass) {
        ret str::concat(~[copy userinfo.user, ~":",
                          option::unwrap(copy userinfo.pass),
                          ~"@"]);
    } else {
        ret str::concat(~[copy userinfo.user, ~"@"]);
    }
}

fn query_from_str(rawquery: ~str) -> query {
    let mut query: query = ~[];
    if str::len(rawquery) != 0 {
        for str::split_char(rawquery, '&').each |p| {
            let (k, v) = split_char_first(p, '=');
            vec::push(query, (k, v));
        };
    }
    ret query;
}

fn query_to_str(query: query) -> ~str {
    let mut strvec = ~[];
    for query.each |kv| {
        let (k, v) = kv;
        strvec += ~[#fmt("%s=%s", k, v)];
    };
    ret str::connect(strvec, ~"&");
}

fn get_scheme(rawurl: ~str) -> option::option<(~str, ~str)> {
    for str::each_chari(rawurl) |i,c| {
        if char::is_alphabetic(c) {
            again;
        } else if c == ':' && i != 0 {
            ret option::some((rawurl.slice(0,i),
                              rawurl.slice(i+3,str::len(rawurl))));
        } else {
            ret option::none;
        }
    };
    ret option::none;
}

/**
 * Parse a `str` to a `url`
 *
 * # Arguments
 *
 * `rawurl` - a string representing a full url, including scheme.
 *
 * # Returns
 *
 * a `url` that contains the parsed representation of the url.
 *
 */

fn from_str(rawurl: ~str) -> result::result<url, ~str> {
    let mut schm = get_scheme(rawurl);
    if option::is_none(schm) {
        ret result::err(~"invalid scheme");
    }
    let (scheme, rest) = option::unwrap(schm);
    let (u, rest) = split_char_first(rest, '@');
    let user = if str::len(rest) == 0 {
        option::none
    } else {
        option::some(userinfo_from_str(u))
    };
    let rest = if str::len(rest) == 0 {
         u
    } else {
        rest
    };
    let (rest, frag) = split_char_first(rest, '#');
    let fragment = if str::len(frag) == 0 {
        option::none
    } else {
        option::some(frag)
    };
    let (rest, query) = split_char_first(rest, '?');
    let query = query_from_str(query);
    let (host, pth) = split_char_first(rest, '/');
    let mut path = pth;
    if str::len(path) != 0 {
        str::unshift_char(path, '/');
    }

    ret result::ok(url(scheme, user, host, path, query, fragment));
}

/**
 * Format a `url` as a string
 *
 * # Arguments
 *
 * `url` - a url.
 *
 * # Returns
 *
 * a `str` that contains the formatted url. Note that this will usually
 * be an inverse of `from_str` but might strip out unneeded separators.
 * for example, "http://somehost.com?", when parsed and formatted, will
 * result in just "http://somehost.com".
 *
 */
fn to_str(url: url) -> ~str {
    let user = if option::is_some(url.user) {
      userinfo_to_str(option::unwrap(copy url.user))
    } else {
       ~""
    };
    let query = if url.query.len() == 0 {
        ~""
    } else {
        str::concat(~[~"?", query_to_str(url.query)])
    };
    let fragment = if option::is_some(url.fragment) {
        str::concat(~[~"#", option::unwrap(copy url.fragment)])
    } else {
        ~""
    };

    ret str::concat(~[copy url.scheme,
                      ~"://",
                      user,
                      copy url.host,
                      copy url.path,
                      query,
                      fragment]);
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_full_url_parse_and_format() {
        let url = ~"http://user:pass@rust-lang.org/doc?s=v#something";
        assert to_str(result::unwrap(from_str(url))) == url;
    }

    #[test]
    fn test_userless_url_parse_and_format() {
        let url = ~"http://rust-lang.org/doc?s=v#something";
        assert to_str(result::unwrap(from_str(url))) == url;
    }

    #[test]
    fn test_queryless_url_parse_and_format() {
        let url = ~"http://user:pass@rust-lang.org/doc#something";
        assert to_str(result::unwrap(from_str(url))) == url;
    }

    #[test]
    fn test_empty_query_url_parse_and_format() {
        let url = ~"http://user:pass@rust-lang.org/doc?#something";
        let should_be = ~"http://user:pass@rust-lang.org/doc#something";
        assert to_str(result::unwrap(from_str(url))) == should_be;
    }

    #[test]
    fn test_fragmentless_url_parse_and_format() {
        let url = ~"http://user:pass@rust-lang.org/doc?q=v";
        assert to_str(result::unwrap(from_str(url))) == url;
    }

    #[test]
    fn test_minimal_url_parse_and_format() {
        let url = ~"http://rust-lang.org/doc";
        assert to_str(result::unwrap(from_str(url))) == url;
    }

    #[test]
    fn test_scheme_host_only_url_parse_and_format() {
        let url = ~"http://rust-lang.org";
        assert to_str(result::unwrap(from_str(url))) == url;
    }

    #[test]
    fn test_pathless_url_parse_and_format() {
        let url = ~"http://user:pass@rust-lang.org?q=v#something";
        assert to_str(result::unwrap(from_str(url))) == url;
    }

    #[test]
    fn test_scheme_host_fragment_only_url_parse_and_format() {
        let url = ~"http://rust-lang.org#something";
        assert to_str(result::unwrap(from_str(url))) == url;
    }

}
