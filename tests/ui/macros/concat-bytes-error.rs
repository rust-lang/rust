//@ edition: 2021
// 2021 edition for C string literals

#![feature(concat_bytes)]

fn main() {
    // Identifiers
    concat_bytes!(pie); //~ ERROR expected a byte literal
    concat_bytes!(pie, pie); //~ ERROR expected a byte literal

    // String literals
    concat_bytes!("tnrsi", "tnri"); //~ ERROR cannot concatenate string literals
    //~^ SUGGESTION b"tnrsi"
    concat_bytes!(r"tnrsi", r"tnri"); //~ ERROR cannot concatenate string literals
    //~^ SUGGESTION br"tnrsi"
    concat_bytes!(r#"tnrsi"#, r###"tnri"###); //~ ERROR cannot concatenate string literals
    //~^ SUGGESTION br#"tnrsi"#
    concat_bytes!(c"tnrsi", c"tnri"); //~ ERROR cannot concatenate C string literals
    //~^ SUGGESTION b"tnrsi\0"
    concat_bytes!(cr"tnrsi", cr"tnri"); //~ ERROR cannot concatenate C string literals
    concat_bytes!(cr#"tnrsi"#, cr###"tnri"###); //~ ERROR cannot concatenate C string literals

    // Other literals
    concat_bytes!(2.8); //~ ERROR cannot concatenate float literals
    concat_bytes!(300); //~ ERROR cannot concatenate numeric literals
    //~^ SUGGESTION [300]
    concat_bytes!('a'); //~ ERROR cannot concatenate character literals
    //~^ SUGGESTION b'a'
    concat_bytes!(true, false); //~ ERROR cannot concatenate boolean literals
    concat_bytes!(42, b"va", b'l'); //~ ERROR cannot concatenate numeric literals
    //~^ SUGGESTION [42]
    concat_bytes!(42, b"va", b'l', [1, 2]); //~ ERROR cannot concatenate numeric literals
    //~^ SUGGESTION [42]

    // Nested items
    concat_bytes!([
        "hi", //~ ERROR cannot concatenate string literals
    ]);
    concat_bytes!([
        'a', //~ ERROR cannot concatenate character literals
        //~^ SUGGESTION b'a'
    ]);
    concat_bytes!([
        true, //~ ERROR cannot concatenate boolean literals
    ]);
    concat_bytes!([
        false, //~ ERROR cannot concatenate boolean literals
    ]);
    concat_bytes!([
        2.6, //~ ERROR cannot concatenate float literals
    ]);
    concat_bytes!([
        265, //~ ERROR numeric literal is out of bounds
    ]);
    concat_bytes!([
        -33, //~ ERROR expected a byte literal
    ]);
    concat_bytes!([
        b"hi!", //~ ERROR cannot concatenate doubly nested array
    ]);
    concat_bytes!([
        [5, 6, 7], //~ ERROR cannot concatenate doubly nested array
    ]);
    concat_bytes!(5u16); //~ ERROR cannot concatenate numeric literals
    //~^ SUGGESTION [5u16]
    concat_bytes!([5u16]); //~ ERROR numeric literal is not a `u8`
    concat_bytes!([3; ()]); //~ ERROR repeat count is not a positive number
    concat_bytes!([3; -2]); //~ ERROR repeat count is not a positive number
    concat_bytes!([pie; -2]); //~ ERROR repeat count is not a positive number
    concat_bytes!([pie; 2]); //~ ERROR expected a byte literal
    concat_bytes!([2.2; 0]); //~ ERROR cannot concatenate float literals
    concat_bytes!([5.5; ()]); //~ ERROR repeat count is not a positive number
    concat_bytes!([[1, 2, 3]; 3]); //~ ERROR cannot concatenate doubly nested array
    concat_bytes!([[42; 2]; 3]); //~ ERROR cannot concatenate doubly nested array
}
