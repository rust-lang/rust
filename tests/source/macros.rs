itemmacro!(this, is.not() .formatted(yet));

fn main() {
    foo! ( );

    bar!( a , b , c );

    baz!(1+2+3, quux. kaas());

    quux!(AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA, BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB);

    kaas!(/* comments */ a /* post macro */, b /* another */);

    trailingcomma!( a , b , c , );

    noexpr!( i am not an expression, OK? );

    vec! [ a , b , c];

    vec! [AAAAAA, AAAAAA, AAAAAA, AAAAAA, AAAAAA, AAAAAA, AAAAAA, AAAAAA, AAAAAA,
          BBBBB, 5, 100-30, 1.33, b, b, b];

    vec! [a /* comment */];

    foo(makro!(1,   3));

    hamkaas!{ () };

    macrowithbraces! {dont,    format, me}

    x!(fn);
}
