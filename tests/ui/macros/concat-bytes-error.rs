#![feature(concat_bytes)]

fn main() {
    concat_bytes!(pie); //~ ERROR expected a byte literal
    concat_bytes!(pie, pie); //~ ERROR expected a byte literal
    concat_bytes!("tnrsi", "tnri"); //~ ERROR cannot concatenate string literals
    concat_bytes!(2.8); //~ ERROR cannot concatenate float literals
    concat_bytes!(300); //~ ERROR cannot concatenate numeric literals
    concat_bytes!('a'); //~ ERROR cannot concatenate character literals
    concat_bytes!(true, false); //~ ERROR cannot concatenate boolean literals
    concat_bytes!(42, b"va", b'l'); //~ ERROR cannot concatenate numeric literals
    concat_bytes!(42, b"va", b'l', [1, 2]); //~ ERROR cannot concatenate numeric literals
    concat_bytes!([
        "hi", //~ ERROR cannot concatenate string literals
    ]);
    concat_bytes!([
        'a', //~ ERROR cannot concatenate character literals
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
