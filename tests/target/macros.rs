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

    foo!(#[doc = "bar"] baz);
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

// Put the following tests with macro invocations whose arguments cannot be parsed as expressioins
// at the end of the file for now.

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
