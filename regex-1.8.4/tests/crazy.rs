mat!(ascii_literal, r"a", "a", Some((0, 1)));

// Some crazy expressions from regular-expressions.info.
mat!(
    match_ranges,
    r"(?-u)\b(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\b",
    "num: 255",
    Some((5, 8))
);
mat!(
    match_ranges_not,
    r"(?-u)\b(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\b",
    "num: 256",
    None
);
mat!(match_float1, r"[-+]?[0-9]*\.?[0-9]+", "0.1", Some((0, 3)));
mat!(match_float2, r"[-+]?[0-9]*\.?[0-9]+", "0.1.2", Some((0, 3)));
mat!(match_float3, r"[-+]?[0-9]*\.?[0-9]+", "a1.2", Some((1, 4)));
mat!(match_float4, r"^[-+]?[0-9]*\.?[0-9]+$", "1.a", None);
mat!(
    match_email,
    r"(?i-u)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}\b",
    "mine is jam.slam@gmail.com ",
    Some((8, 26))
);
mat!(
    match_email_not,
    r"(?i-u)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}\b",
    "mine is jam.slam@gmail ",
    None
);
mat!(
    match_email_big,
    r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
    "mine is jam.slam@gmail.com ",
    Some((8, 26))
);
mat!(
    match_date1,
    r"(?-u)^(19|20)\d\d[- /.](0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])$",
    "1900-01-01",
    Some((0, 10))
);
mat!(
    match_date2,
    r"(?-u)^(19|20)\d\d[- /.](0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])$",
    "1900-00-01",
    None
);
mat!(
    match_date3,
    r"(?-u)^(19|20)\d\d[- /.](0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])$",
    "1900-13-01",
    None
);

// Do some crazy dancing with the start/end assertions.
matiter!(match_start_end_empty, r"^$", "", (0, 0));
matiter!(match_start_end_empty_many_1, r"^$^$^$", "", (0, 0));
matiter!(match_start_end_empty_many_2, r"^^^$$$", "", (0, 0));
matiter!(match_start_end_empty_rev, r"$^", "", (0, 0));
matiter!(
    match_start_end_empty_rep,
    r"(?:^$)*",
    "a\nb\nc",
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5)
);
matiter!(
    match_start_end_empty_rep_rev,
    r"(?:$^)*",
    "a\nb\nc",
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5)
);

// Test negated character classes.
mat!(negclass_letters, r"[^ac]", "acx", Some((2, 3)));
mat!(negclass_letter_comma, r"[^a,]", "a,x", Some((2, 3)));
mat!(negclass_letter_space, r"[^a[:space:]]", "a x", Some((2, 3)));
mat!(negclass_comma, r"[^,]", ",,x", Some((2, 3)));
mat!(negclass_space, r"[^[:space:]]", " a", Some((1, 2)));
mat!(negclass_space_comma, r"[^,[:space:]]", ", a", Some((2, 3)));
mat!(negclass_comma_space, r"[^[:space:],]", " ,a", Some((2, 3)));
mat!(negclass_ascii, r"[^[:alpha:]Z]", "A1", Some((1, 2)));

// Test that repeated empty expressions don't loop forever.
mat!(lazy_many_many, r"((?:.*)*?)=", "a=b", Some((0, 2)));
mat!(lazy_many_optional, r"((?:.?)*?)=", "a=b", Some((0, 2)));
mat!(lazy_one_many_many, r"((?:.*)+?)=", "a=b", Some((0, 2)));
mat!(lazy_one_many_optional, r"((?:.?)+?)=", "a=b", Some((0, 2)));
mat!(lazy_range_min_many, r"((?:.*){1,}?)=", "a=b", Some((0, 2)));
mat!(lazy_range_many, r"((?:.*){1,2}?)=", "a=b", Some((0, 2)));
mat!(greedy_many_many, r"((?:.*)*)=", "a=b", Some((0, 2)));
mat!(greedy_many_optional, r"((?:.?)*)=", "a=b", Some((0, 2)));
mat!(greedy_one_many_many, r"((?:.*)+)=", "a=b", Some((0, 2)));
mat!(greedy_one_many_optional, r"((?:.?)+)=", "a=b", Some((0, 2)));
mat!(greedy_range_min_many, r"((?:.*){1,})=", "a=b", Some((0, 2)));
mat!(greedy_range_many, r"((?:.*){1,2})=", "a=b", Some((0, 2)));

// Test that we handle various flavors of empty expressions.
matiter!(match_empty1, r"", "", (0, 0));
matiter!(match_empty2, r"", "abc", (0, 0), (1, 1), (2, 2), (3, 3));
matiter!(match_empty3, r"()", "abc", (0, 0), (1, 1), (2, 2), (3, 3));
matiter!(match_empty4, r"()*", "abc", (0, 0), (1, 1), (2, 2), (3, 3));
matiter!(match_empty5, r"()+", "abc", (0, 0), (1, 1), (2, 2), (3, 3));
matiter!(match_empty6, r"()?", "abc", (0, 0), (1, 1), (2, 2), (3, 3));
matiter!(match_empty7, r"()()", "abc", (0, 0), (1, 1), (2, 2), (3, 3));
matiter!(match_empty8, r"()+|z", "abc", (0, 0), (1, 1), (2, 2), (3, 3));
matiter!(match_empty9, r"z|()+", "abc", (0, 0), (1, 1), (2, 2), (3, 3));
matiter!(match_empty10, r"()+|b", "abc", (0, 0), (1, 1), (2, 2), (3, 3));
matiter!(match_empty11, r"b|()+", "abc", (0, 0), (1, 2), (3, 3));
matiter!(match_empty12, r"|b", "abc", (0, 0), (1, 1), (2, 2), (3, 3));
matiter!(match_empty13, r"b|", "abc", (0, 0), (1, 2), (3, 3));
matiter!(match_empty14, r"|z", "abc", (0, 0), (1, 1), (2, 2), (3, 3));
matiter!(match_empty15, r"z|", "abc", (0, 0), (1, 1), (2, 2), (3, 3));
matiter!(match_empty16, r"|", "abc", (0, 0), (1, 1), (2, 2), (3, 3));
matiter!(match_empty17, r"||", "abc", (0, 0), (1, 1), (2, 2), (3, 3));
matiter!(match_empty18, r"||z", "abc", (0, 0), (1, 1), (2, 2), (3, 3));
matiter!(match_empty19, r"(?:)|b", "abc", (0, 0), (1, 1), (2, 2), (3, 3));
matiter!(match_empty20, r"b|(?:)", "abc", (0, 0), (1, 2), (3, 3));
matiter!(match_empty21, r"(?:|)", "abc", (0, 0), (1, 1), (2, 2), (3, 3));
matiter!(match_empty22, r"(?:|)|z", "abc", (0, 0), (1, 1), (2, 2), (3, 3));
matiter!(match_empty23, r"a(?:)|b", "abc", (0, 1), (1, 2));

// Test that the DFA can handle pathological cases.
// (This should result in the DFA's cache being flushed too frequently, which
// should cause it to quit and fall back to the NFA algorithm.)
#[test]
fn dfa_handles_pathological_case() {
    fn ones_and_zeroes(count: usize) -> String {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        let mut rng = SmallRng::from_entropy();
        let mut s = String::new();
        for _ in 0..count {
            if rng.gen() {
                s.push('1');
            } else {
                s.push('0');
            }
        }
        s
    }

    let re = regex!(r"[01]*1[01]{20}$");
    let text = {
        let mut pieces = ones_and_zeroes(100_000);
        pieces.push('1');
        pieces.push_str(&ones_and_zeroes(20));
        pieces
    };
    assert!(re.is_match(text!(&*text)));
}

#[test]
fn nest_limit_makes_it_parse() {
    use regex::RegexBuilder;

    RegexBuilder::new(
        r#"(?-u)
        2(?:
          [45]\d{3}|
          7(?:
            1[0-267]|
            2[0-289]|
            3[0-29]|
            4[01]|
            5[1-3]|
            6[013]|
            7[0178]|
            91
          )|
          8(?:
            0[125]|
            [139][1-6]|
            2[0157-9]|
            41|
            6[1-35]|
            7[1-5]|
            8[1-8]|
            90
          )|
          9(?:
            0[0-2]|
            1[0-4]|
            2[568]|
            3[3-6]|
            5[5-7]|
            6[0167]|
            7[15]|
            8[0146-9]
          )
        )\d{4}|
        3(?:
          12?[5-7]\d{2}|
          0(?:
            2(?:
              [025-79]\d|
              [348]\d{1,2}
            )|
            3(?:
              [2-4]\d|
              [56]\d?
            )
          )|
          2(?:
            1\d{2}|
            2(?:
              [12]\d|
              [35]\d{1,2}|
              4\d?
            )
          )|
          3(?:
            1\d{2}|
            2(?:
              [2356]\d|
              4\d{1,2}
            )
          )|
          4(?:
            1\d{2}|
            2(?:
              2\d{1,2}|
              [47]|
              5\d{2}
            )
          )|
          5(?:
            1\d{2}|
            29
          )|
          [67]1\d{2}|
          8(?:
            1\d{2}|
            2(?:
              2\d{2}|
              3|
              4\d
            )
          )
        )\d{3}|
        4(?:
          0(?:
            2(?:
              [09]\d|
              7
            )|
            33\d{2}
          )|
          1\d{3}|
          2(?:
            1\d{2}|
            2(?:
              [25]\d?|
              [348]\d|
              [67]\d{1,2}
            )
          )|
          3(?:
            1\d{2}(?:
              \d{2}
            )?|
            2(?:
              [045]\d|
              [236-9]\d{1,2}
            )|
            32\d{2}
          )|
          4(?:
            [18]\d{2}|
            2(?:
              [2-46]\d{2}|
              3
            )|
            5[25]\d{2}
          )|
          5(?:
            1\d{2}|
            2(?:
              3\d|
              5
            )
          )|
          6(?:
            [18]\d{2}|
            2(?:
              3(?:
                \d{2}
              )?|
              [46]\d{1,2}|
              5\d{2}|
              7\d
            )|
            5(?:
              3\d?|
              4\d|
              [57]\d{1,2}|
              6\d{2}|
              8
            )
          )|
          71\d{2}|
          8(?:
            [18]\d{2}|
            23\d{2}|
            54\d{2}
          )|
          9(?:
            [18]\d{2}|
            2[2-5]\d{2}|
            53\d{1,2}
          )
        )\d{3}|
        5(?:
          02[03489]\d{2}|
          1\d{2}|
          2(?:
            1\d{2}|
            2(?:
              2(?:
                \d{2}
              )?|
              [457]\d{2}
            )
          )|
          3(?:
            1\d{2}|
            2(?:
              [37](?:
                \d{2}
              )?|
              [569]\d{2}
            )
          )|
          4(?:
            1\d{2}|
            2[46]\d{2}
          )|
          5(?:
            1\d{2}|
            26\d{1,2}
          )|
          6(?:
            [18]\d{2}|
            2|
            53\d{2}
          )|
          7(?:
            1|
            24
          )\d{2}|
          8(?:
            1|
            26
          )\d{2}|
          91\d{2}
        )\d{3}|
        6(?:
          0(?:
            1\d{2}|
            2(?:
              3\d{2}|
              4\d{1,2}
            )
          )|
          2(?:
            2[2-5]\d{2}|
            5(?:
              [3-5]\d{2}|
              7
            )|
            8\d{2}
          )|
          3(?:
            1|
            2[3478]
          )\d{2}|
          4(?:
            1|
            2[34]
          )\d{2}|
          5(?:
            1|
            2[47]
          )\d{2}|
          6(?:
            [18]\d{2}|
            6(?:
              2(?:
                2\d|
                [34]\d{2}
              )|
              5(?:
                [24]\d{2}|
                3\d|
                5\d{1,2}
              )
            )
          )|
          72[2-5]\d{2}|
          8(?:
            1\d{2}|
            2[2-5]\d{2}
          )|
          9(?:
            1\d{2}|
            2[2-6]\d{2}
          )
        )\d{3}|
        7(?:
          (?:
            02|
            [3-589]1|
            6[12]|
            72[24]
          )\d{2}|
          21\d{3}|
          32
        )\d{3}|
        8(?:
          (?:
            4[12]|
            [5-7]2|
            1\d?
          )|
          (?:
            0|
            3[12]|
            [5-7]1|
            217
          )\d
        )\d{4}|
        9(?:
          [35]1|
          (?:
            [024]2|
            81
          )\d|
          (?:
            1|
            [24]1
          )\d{2}
        )\d{3}
        "#,
    )
    .build()
    .unwrap();
}
