// rustfmt-normalize_comments: true
itemmacro!(this, is.now().formatted(yay));

itemmacro!(
    really,
    long.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbb()
        .is
        .formatted()
);

itemmacro!{this, is.bracket().formatted()}

peg_file! modname("mygrammarfile.rustpeg");

fn main() {
    foo!();

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

    hamkaas!{ () };

    macrowithbraces! {dont,    format, me}

    x!(fn);

    some_macro!();

    some_macro![];

    some_macro!{
        // comment
    };

    some_macro!{
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
}

impl X {
    empty_invoc!{}
}

fn issue_1279() {
    println!("dsfs"); // a comment
}

fn issue_1555() {
    let hello = &format!(
        "HTTP/1.1 200 OK\r\nServer: {}\r\n\r\n{}",
        "65454654654654654654654655464",
        "4"
    );
}

fn issue1178() {
    macro_rules! foo {
        (#[$attr:meta] $name:ident) => {}
    }

    foo!(
        #[doc = "bar"]
        baz
    );
}

fn issue1739() {
    sql_function!(
        add_rss_item,
        add_rss_item_t,
        (
            a: types::Integer,
            b: types::Timestamptz,
            c: types::Text,
            d: types::Text,
            e: types::Text
        )
    );

    w.slice_mut(s![
        ..,
        init_size[1] - extreeeeeeeeeeeeeeeeeeeeeeeem..init_size[1],
        ..
    ]).par_map_inplace(|el| *el = 0.);
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
        }
    }
}

// #1577
fn issue1577() {
    let json = json!({
        "foo": "bar",
    });
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
    ($expr:expr) => (match $expr {
        Some(val) => val,

        None => { return None; }
    })
}
