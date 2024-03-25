#[test]
fn empty_regex_empty_match() {
    let re = regex!("");
    assert_eq!(vec![(0, 0)], findall!(re, ""));
}

#[test]
fn empty_regex_nonempty_match() {
    let re = regex!("");
    assert_eq!(vec![(0, 0), (1, 1), (2, 2), (3, 3)], findall!(re, "abc"));
}

#[test]
fn one_zero_length_match() {
    let re = regex!(r"[0-9]*");
    assert_eq!(vec![(0, 0), (1, 2), (3, 4)], findall!(re, "a1b2"));
}

#[test]
fn many_zero_length_match() {
    let re = regex!(r"[0-9]*");
    assert_eq!(
        vec![(0, 0), (1, 2), (3, 3), (4, 4), (5, 6)],
        findall!(re, "a1bbb2")
    );
}

#[test]
fn many_sequential_zero_length_match() {
    let re = regex!(r"[0-9]?");
    assert_eq!(
        vec![(0, 0), (1, 2), (2, 3), (4, 5), (6, 6)],
        findall!(re, "a12b3c")
    );
}

#[test]
fn quoted_bracket_set() {
    let re = regex!(r"([\x{5b}\x{5d}])");
    assert_eq!(vec![(0, 1), (1, 2)], findall!(re, "[]"));
    let re = regex!(r"([\[\]])");
    assert_eq!(vec![(0, 1), (1, 2)], findall!(re, "[]"));
}

#[test]
fn first_range_starts_with_left_bracket() {
    let re = regex!(r"([\[-z])");
    assert_eq!(vec![(0, 1), (1, 2)], findall!(re, "[]"));
}

#[test]
fn range_ends_with_escape() {
    let re = regex!(r"([\[-\x{5d}])");
    assert_eq!(vec![(0, 1), (1, 2)], findall!(re, "[]"));
}

#[test]
fn empty_match_find_iter() {
    let re = regex!(r".*?");
    assert_eq!(vec![(0, 0), (1, 1), (2, 2), (3, 3)], findall!(re, "abc"));
}

#[test]
fn empty_match_captures_iter() {
    let re = regex!(r".*?");
    let ms: Vec<_> = re
        .captures_iter(text!("abc"))
        .map(|c| c.get(0).unwrap())
        .map(|m| (m.start(), m.end()))
        .collect();
    assert_eq!(ms, vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
}

#[test]
fn capture_names() {
    let re = regex!(r"(.)(?P<a>.)");
    assert_eq!(3, re.captures_len());
    assert_eq!((3, Some(3)), re.capture_names().size_hint());
    assert_eq!(
        vec![None, None, Some("a")],
        re.capture_names().collect::<Vec<_>>()
    );
}

#[test]
fn regex_string() {
    assert_eq!(r"[a-zA-Z0-9]+", regex!(r"[a-zA-Z0-9]+").as_str());
    assert_eq!(r"[a-zA-Z0-9]+", &format!("{}", regex!(r"[a-zA-Z0-9]+")));
    assert_eq!(r"[a-zA-Z0-9]+", &format!("{:?}", regex!(r"[a-zA-Z0-9]+")));
}

#[test]
fn capture_index() {
    let re = regex!(r"^(?P<name>.+)$");
    let cap = re.captures(t!("abc")).unwrap();
    assert_eq!(&cap[0], t!("abc"));
    assert_eq!(&cap[1], t!("abc"));
    assert_eq!(&cap["name"], t!("abc"));
}

#[test]
#[should_panic]
#[cfg_attr(all(target_env = "msvc", target_pointer_width = "32"), ignore)]
fn capture_index_panic_usize() {
    let re = regex!(r"^(?P<name>.+)$");
    let cap = re.captures(t!("abc")).unwrap();
    let _ = cap[2];
}

#[test]
#[should_panic]
#[cfg_attr(all(target_env = "msvc", target_pointer_width = "32"), ignore)]
fn capture_index_panic_name() {
    let re = regex!(r"^(?P<name>.+)$");
    let cap = re.captures(t!("abc")).unwrap();
    let _ = cap["bad name"];
}

#[test]
fn capture_index_lifetime() {
    // This is a test of whether the types on `caps["..."]` are general
    // enough. If not, this will fail to typecheck.
    fn inner(s: &str) -> usize {
        let re = regex!(r"(?P<number>[0-9]+)");
        let caps = re.captures(t!(s)).unwrap();
        caps["number"].len()
    }
    assert_eq!(3, inner("123"));
}

#[test]
fn capture_misc() {
    let re = regex!(r"(.)(?P<a>a)?(.)(?P<b>.)");
    let cap = re.captures(t!("abc")).unwrap();

    assert_eq!(5, cap.len());

    assert_eq!((0, 3), {
        let m = cap.get(0).unwrap();
        (m.start(), m.end())
    });
    assert_eq!(None, cap.get(2));
    assert_eq!((2, 3), {
        let m = cap.get(4).unwrap();
        (m.start(), m.end())
    });

    assert_eq!(t!("abc"), match_text!(cap.get(0).unwrap()));
    assert_eq!(None, cap.get(2));
    assert_eq!(t!("c"), match_text!(cap.get(4).unwrap()));

    assert_eq!(None, cap.name("a"));
    assert_eq!(t!("c"), match_text!(cap.name("b").unwrap()));
}

#[test]
fn sub_capture_matches() {
    let re = regex!(r"([a-z])(([a-z])|([0-9]))");
    let cap = re.captures(t!("a5")).unwrap();
    let subs: Vec<_> = cap.iter().collect();

    assert_eq!(5, subs.len());
    assert!(subs[0].is_some());
    assert!(subs[1].is_some());
    assert!(subs[2].is_some());
    assert!(subs[3].is_none());
    assert!(subs[4].is_some());

    assert_eq!(t!("a5"), match_text!(subs[0].unwrap()));
    assert_eq!(t!("a"), match_text!(subs[1].unwrap()));
    assert_eq!(t!("5"), match_text!(subs[2].unwrap()));
    assert_eq!(t!("5"), match_text!(subs[4].unwrap()));
}

expand!(expand1, r"(?-u)(?P<foo>\w+)", "abc", "$foo", "abc");
expand!(expand2, r"(?-u)(?P<foo>\w+)", "abc", "$0", "abc");
expand!(expand3, r"(?-u)(?P<foo>\w+)", "abc", "$1", "abc");
expand!(expand4, r"(?-u)(?P<foo>\w+)", "abc", "$$1", "$1");
expand!(expand5, r"(?-u)(?P<foo>\w+)", "abc", "$$foo", "$foo");
expand!(expand6, r"(?-u)(?P<a>\w+)\s+(?P<b>\d+)", "abc 123", "$b$a", "123abc");
expand!(expand7, r"(?-u)(?P<a>\w+)\s+(?P<b>\d+)", "abc 123", "z$bz$az", "z");
expand!(
    expand8,
    r"(?-u)(?P<a>\w+)\s+(?P<b>\d+)",
    "abc 123",
    ".$b.$a.",
    ".123.abc."
);
expand!(
    expand9,
    r"(?-u)(?P<a>\w+)\s+(?P<b>\d+)",
    "abc 123",
    " $b $a ",
    " 123 abc "
);
expand!(expand10, r"(?-u)(?P<a>\w+)\s+(?P<b>\d+)", "abc 123", "$bz$az", "");

expand!(expand_name1, r"%(?P<Z>[a-z]+)", "%abc", "$Z%", "abc%");
expand!(expand_name2, r"\[(?P<Z>[a-z]+)", "[abc", "$Z[", "abc[");
expand!(expand_name3, r"\{(?P<Z>[a-z]+)", "{abc", "$Z{", "abc{");
expand!(expand_name4, r"\}(?P<Z>[a-z]+)", "}abc", "$Z}", "abc}");
expand!(expand_name5, r"%([a-z]+)", "%abc", "$1a%", "%");
expand!(expand_name6, r"%([a-z]+)", "%abc", "${1}a%", "abca%");
expand!(expand_name7, r"\[(?P<Z[>[a-z]+)", "[abc", "${Z[}[", "abc[");
expand!(expand_name8, r"\[(?P<Z[>[a-z]+)", "[abc", "${foo}[", "[");
expand!(expand_name9, r"\[(?P<Z[>[a-z]+)", "[abc", "${1a}[", "[");
expand!(expand_name10, r"\[(?P<Z[>[a-z]+)", "[abc", "${#}[", "[");
expand!(expand_name11, r"\[(?P<Z[>[a-z]+)", "[abc", "${$$}[", "[");

split!(
    split1,
    r"(?-u)\s+",
    "a b\nc\td\n\t e",
    &[t!("a"), t!("b"), t!("c"), t!("d"), t!("e")]
);
split!(
    split2,
    r"(?-u)\b",
    "a b c",
    &[t!(""), t!("a"), t!(" "), t!("b"), t!(" "), t!("c"), t!("")]
);
split!(split3, r"a$", "a", &[t!(""), t!("")]);
split!(split_none, r"-", r"a", &[t!("a")]);
split!(split_trailing_blank, r"-", r"a-", &[t!("a"), t!("")]);
split!(split_trailing_blanks, r"-", r"a--", &[t!("a"), t!(""), t!("")]);
split!(split_empty, r"-", r"", &[t!("")]);

splitn!(splitn_below_limit, r"-", r"a", 2, &[t!("a")]);
splitn!(splitn_at_limit, r"-", r"a-b", 2, &[t!("a"), t!("b")]);
splitn!(splitn_above_limit, r"-", r"a-b-c", 2, &[t!("a"), t!("b-c")]);
splitn!(splitn_zero_limit, r"-", r"a-b", 0, empty_vec!());
splitn!(splitn_trailing_blank, r"-", r"a-", 2, &[t!("a"), t!("")]);
splitn!(splitn_trailing_separator, r"-", r"a--", 2, &[t!("a"), t!("-")]);
splitn!(splitn_empty, r"-", r"", 1, &[t!("")]);
