// rustfmt-normalize_comments: true
// rustfmt-format_macro_matchers: true
itemmacro!(this, is.now().formatted(yay));

itemmacro!(
    really,
    long.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbb()
        .is
        .formatted()
);

itemmacro! {this, is.brace().formatted()}

fn main() {
    foo!();

    foo!(,);

    bar!(a, b, c);

    bar!(a, b, c,);

    baz!(1 + 2 + 3, quux.kaas());

    quux!(
        AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,
        BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
    );

    kaas!(
        // comments
        a, // post macro
        b  // another
    );

    trailingcomma!(a, b, c,);
    // Preserve trailing comma only when necessary.
    ok!(file.seek(SeekFrom::Start(
        table.map(|table| fixture.offset(table)).unwrap_or(0),
    )));

    noexpr!( i am not an expression, OK? );

    vec![a, b, c];

    vec![
        AAAAAA,
        AAAAAA,
        AAAAAA,
        AAAAAA,
        AAAAAA,
        AAAAAA,
        AAAAAA,
        AAAAAA,
        AAAAAA,
        BBBBB,
        5,
        100 - 30,
        1.33,
        b,
        b,
        b,
    ];

    vec![a /* comment */];

    // Trailing spaces after a comma
    vec![a];

    vec![a; b];
    vec![a; b];
    vec![a; b];

    vec![a, b; c];
    vec![a; b, c];

    vec![
        a;
        (|x| {
            let y = x + 1;
            let z = y + 1;
            z
        })(2)
    ];
    vec![
        a;
        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ];
    vec![a; unsafe { x + 1 }];

    unknown_bracket_macro__comma_should_not_be_stripped![a,];

    foo(makro!(1, 3));

    hamkaas! { () };

    macrowithbraces! {dont,    format, me}

    x!(fn);

    some_macro!();

    some_macro![];

    some_macro! {
        // comment
    };

    some_macro! {
        // comment
    };

    some_macro!(
        // comment
        not function like
    );

    // #1712
    let image = gray_image!(
        00, 01, 02;
        10, 11, 12;
        20, 21, 22);

    // #1092
    chain!(input, a:take!(max_size), || []);

    // #2727
    foo!("bar");
}

impl X {
    empty_invoc! {}
    empty_invoc! {}
}

fn issue_1279() {
    println!("dsfs"); // a comment
}

fn issue_1555() {
    let hello = &format!(
        "HTTP/1.1 200 OK\r\nServer: {}\r\n\r\n{}",
        "65454654654654654654654655464", "4"
    );
}

fn issue1178() {
    macro_rules! foo {
        (#[$attr:meta] $name:ident) => {};
    }

    foo!(
        #[doc = "bar"]
        baz
    );
}

fn issue1739() {
    sql_function!(add_rss_item,
                  add_rss_item_t,
                  (a: types::Integer,
                   b: types::Timestamptz,
                   c: types::Text,
                   d: types::Text,
                   e: types::Text));

    w.slice_mut(s![
        ..,
        init_size[1] - extreeeeeeeeeeeeeeeeeeeeeeeem..init_size[1],
        ..
    ])
    .par_map_inplace(|el| *el = 0.);
}

fn issue_1885() {
    let threads = people
        .into_iter()
        .map(|name| {
            chan_select! {
                rx.recv() => {}
            }
        })
        .collect::<Vec<_>>();
}

fn issue_1917() {
    mod x {
        quickcheck! {
            fn test(a: String, s: String, b: String) -> TestResult {
                if a.find(&s).is_none() {

                    TestResult::from_bool(true)
                } else {
                    TestResult::discard()
                }
            }
        }
    }
}

fn issue_1921() {
    // Macro with tabs.
    lazy_static! {
        static ref ONE: u32 = 1;
        static ref TWO: u32 = 2;
        static ref THREE: u32 = 3;
        static ref FOUR: u32 = {
            let mut acc = 1;
            acc += 1;
            acc += 2;
            acc
        };
    }
}

// #1577
fn issue1577() {
    let json = json!({
        "foo": "bar",
    });
}

// #3174
fn issue_3174() {
    let data = if let Some(debug) = error.debug_info() {
        json!({
            "errorKind": format!("{:?}", error.err_kind()),
            "debugMessage": debug.message,
        })
    } else {
        json!({"errorKind": format!("{:?}", error.err_kind())})
    };
}

gfx_pipeline!(pipe {
    vbuf: gfx::VertexBuffer<Vertex> = (),
    out: gfx::RenderTarget<ColorFormat> = "Target0",
});

// #1919
#[test]
fn __bindgen_test_layout_HandleWithDtor_open0_int_close0_instantiation() {
    assert_eq!(
        ::std::mem::size_of::<HandleWithDtor<::std::os::raw::c_int>>(),
        8usize,
        concat!(
            "Size of template specialization: ",
            stringify!(HandleWithDtor<::std::os::raw::c_int>)
        )
    );
    assert_eq!(
        ::std::mem::align_of::<HandleWithDtor<::std::os::raw::c_int>>(),
        8usize,
        concat!(
            "Alignment of template specialization: ",
            stringify!(HandleWithDtor<::std::os::raw::c_int>)
        )
    );
}

// #878
macro_rules! try_opt {
    ($expr:expr) => {
        match $expr {
            Some(val) => val,

            None => {
                return None;
            }
        }
    };
}

// #2214
// macro call whose argument is an array with trailing comma.
fn issue2214() {
    make_test!(
        str_searcher_ascii_haystack,
        "bb",
        "abbcbbd",
        [
            Reject(0, 1),
            Match(1, 3),
            Reject(3, 4),
            Match(4, 6),
            Reject(6, 7),
        ]
    );
}

fn special_case_macros() {
    let p = eprint!();
    let q = eprint!("{}", 1);
    let r = eprint!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    let s = eprint!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26
    );

    let q = eprintln!("{}", 1);
    let r = eprintln!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    let s = eprintln!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26
    );

    let q = format!("{}", 1);
    let r = format!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    let s = format!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26
    );

    let q = format_args!("{}", 1);
    let r = format_args!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    let s = format_args!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26
    );

    let q = print!("{}", 1);
    let r = print!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    let s = print!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26
    );

    let q = println!("{}", 1);
    let r = println!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    let s = println!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26
    );

    let q = unreachable!("{}", 1);
    let r = unreachable!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    let s = unreachable!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26
    );

    debug!("{}", 1);
    debug!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    debug!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26
    );

    error!("{}", 1);
    error!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    error!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26
    );

    info!("{}", 1);
    info!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    info!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26
    );

    panic!("{}", 1);
    panic!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    panic!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26
    );

    warn!("{}", 1);
    warn!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    warn!(
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26
    );

    assert!();
    assert!(result == 42);
    assert!(result == 42, "Ahoy there, {}!", target);
    assert!(
        result == 42,
        "Arr! While plunderin' the hold, we got '{}' when given '{}' (we expected '{}')",
        result,
        input,
        expected
    );
    assert!(
        result == 42,
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26
    );

    assert_eq!();
    assert_eq!(left);
    assert_eq!(left, right);
    assert_eq!(left, right, "Ahoy there, {}!", target);
    assert_eq!(
        left, right,
        "Arr! While plunderin' the hold, we got '{}' when given '{}' (we expected '{}')",
        result, input, expected
    );
    assert_eq!(
        first_realllllllllllly_long_variable_that_doesnt_fit_one_one_line,
        second_reallllllllllly_long_variable_that_doesnt_fit_one_one_line,
        "Arr! While plunderin' the hold, we got '{}' when given '{}' (we expected '{}')",
        result,
        input,
        expected
    );
    assert_eq!(
        left + 42,
        right,
        "Arr! While plunderin' the hold, we got '{}' when given '{}' (we expected '{}')",
        result,
        input,
        expected
    );
    assert_eq!(
        left,
        right,
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26
    );

    write!(&mut s, "Ahoy there, {}!", target);
    write!(
        &mut s,
        "Arr! While plunderin' the hold, we got '{}' when given '{}' (we expected '{}')",
        result, input, expected
    );
    write!(
        &mut s,
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26
    );

    writeln!(&mut s, "Ahoy there, {}!", target);
    writeln!(
        &mut s,
        "Arr! While plunderin' the hold, we got '{}' when given '{}' (we expected '{}')",
        result, input, expected
    );
    writeln!(
        &mut s,
        "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26
    );
}

// #1209
impl Foo {
    /// foo
    pub fn foo(&self) -> Bar<foo!()> {}
}

// #819
fn macro_in_pattern_position() {
    let x = match y {
        foo!() => (),
        bar!(a, b, c) => (),
        bar!(a, b, c,) => (),
        baz!(1 + 2 + 3, quux.kaas()) => (),
        quux!(
            AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,
            BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
        ) => (),
    };
}

macro foo() {}

pub macro bar($x:ident + $y:expr;) {
    fn foo($x: Foo) {
        long_function(
            a_long_argument_to_a_long_function_is_what_this_is(AAAAAAAAAAAAAAAAAAAAAAAAAAAA),
            $x.bar($y),
        );
    }
}

macro foo() {
    // a comment
    fn foo() {
        // another comment
        bar();
    }
}

// #2574
macro_rules! test {
    () => {{}};
}

macro lex_err($kind: ident $(, $body: expr)*) {
    Err(QlError::LexError(LexError::$kind($($body,)*)))
}

// Preserve trailing comma on item-level macro with `()` or `[]`.
methods![get, post, delete,];
methods!(get, post, delete,);

// #2588
macro_rules! m {
    () => {
        r#"
            test
        "#
    };
}
fn foo() {
    f! {r#"
            test
       "#};
}

// #2591
fn foo() {
    match 0u32 {
        0 => (),
        _ => unreachable!(/* obviously */),
    }
}

fn foo() {
    let _ = column!(/* here */);
}

// #2616
// Preserve trailing comma when using mixed layout for macro call.
fn foo() {
    foo!(
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    );
    foo!(
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    );
}

// #2830
// Preserve trailing comma-less/ness inside nested macro.
named!(
    do_parse_gsv<GsvData>,
    map_res!(
        do_parse!(
            number_of_sentences: map_res!(digit, parse_num::<u16>)
                >> char!(',')
                >> sentence_index: map_res!(digit, parse_num::<u16>)
                >> char!(',')
                >> total_number_of_sats: map_res!(digit, parse_num::<u16>)
                >> char!(',')
                >> sat0: opt!(complete!(parse_gsv_sat_info))
                >> sat1: opt!(complete!(parse_gsv_sat_info))
                >> sat2: opt!(complete!(parse_gsv_sat_info))
                >> sat3: opt!(complete!(parse_gsv_sat_info))
                >> (
                    number_of_sentences,
                    sentence_index,
                    total_number_of_sats,
                    sat0,
                    sat1,
                    sat2,
                    sat3
                )
        ),
        construct_gsv_data
    )
);

// #2857
convert_args!(vec!(1, 2, 3));

// #3031
thread_local!(
    /// TLV Holds a set of JSTraceables that need to be rooted
    static ROOTED_TRACEABLES: RefCell<RootedTraceableSet> = RefCell::new(RootedTraceableSet::new());
);

thread_local![
    /// TLV Holds a set of JSTraceables that need to be rooted
    static ROOTED_TRACEABLES: RefCell<RootedTraceableSet> = RefCell::new(RootedTraceableSet::new());

    /// TLV Holds a set of JSTraceables that need to be rooted
    static ROOTED_TRACEABLES: RefCell<RootedTraceableSet> =
        RefCell::new(RootedTraceableSet::new(0));

    /// TLV Holds a set of JSTraceables that need to be rooted
    static ROOTED_TRACEABLES: RefCell<RootedTraceableSet> =
        RefCell::new(RootedTraceableSet::new(), xxx, yyy);

    /// TLV Holds a set of JSTraceables that need to be rooted
    static ROOTED_TRACEABLES: RefCell<RootedTraceableSet> =
        RefCell::new(RootedTraceableSet::new(1234));
];

fn issue3004() {
    foo!(|_| { () });
    stringify!((foo+));
}

// #3331
pub fn fold_abi<V: Fold + ?Sized>(_visitor: &mut V, _i: Abi) -> Abi {
    Abi {
        extern_token: Token![extern](tokens_helper(_visitor, &_i.extern_token.span)),
        name: (_i.name).map(|it| _visitor.fold_lit_str(it)),
    }
}

// #3463
x! {()}

// #3746
f!(match a {
    4 => &[
        (3, false), // Missing
        (4, true)   // I-frame
    ][..],
});

// #3583
foo!(|x = y|);
