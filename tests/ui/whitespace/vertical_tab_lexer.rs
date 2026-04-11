// ignore-tidy-tab
//
// Tests that the Rust lexer accepts Unicode Pattern_White_Space characters.
//
// Worth noting: the Rust reference defines whitespace as Pattern_White_Space,
// which is not the same as what is_ascii_whitespace or is_whitespace give you.
//
// is_ascii_whitespace follows WhatWG and skips vertical tab (\x0B).
// is_whitespace uses Unicode White_Space, which is a broader set.
//
// The 11 characters that actually count as whitespace in Rust source:
//   \x09 \x0A \x0B \x0C \x0D \x20 \u{85} \u{200E} \u{200F} \u{2028} \u{2029}
//
// Ref: https://github.com/rustfoundation/interop-initiative/issues/53

#[rustfmt::skip]
fn main() {
    // tab (\x09) between `let` and the name
    let	_ws1 = 1_i32;

    // vertical tab (\x0B) between `let` and the name
    // this is the one is_ascii_whitespace gets wrong
    let_ws2 = 2_i32;

    // form feed (\x0C) between `let` and the name
    let_ws3 = 3_i32;

    // plain space (\x20), here just so every character is represented
    let _ws4 = 4_i32;

    // NEL (\u{85}) between `let` and the name
    let_ws5 = 5_i32;

    // left-to-right mark (\u{200E}) between `let` and the name
    let‎_ws6 = 6_i32;

    // right-to-left mark (\u{200F}) between `let` and the name
    let‏_ws7 = 7_i32;

    // \x0A, \x0D, \u{2028}, \u{2029} are also Pattern_White_Space but they
    // act as line endings, so you can't stick them in the middle of a statement.
    // The lexer still handles them correctly at line boundaries.

    // These are Unicode White_Space but NOT Pattern_White_Space, so the Rust
    // lexer won't accept them between tokens:
    //   \u{A0}   no-break space       \u{1680} ogham space mark
    //   \u{2000} en quad              \u{2001} em quad
    //   \u{2002} en space             \u{2003} em space
    //   \u{2004} three-per-em space   \u{2005} four-per-em space
    //   \u{2006} six-per-em space     \u{2007} figure space
    //   \u{2008} punctuation space    \u{2009} thin space
    //   \u{200A} hair space           \u{202F} narrow no-break space
    //   \u{205F} medium math space    \u{3000} ideographic space

    // add them up so the compiler doesn't complain about unused variables
    let _sum = _ws1 + _ws2 + _ws3 + _ws4 + _ws5 + _ws6 + _ws7;
    println!("{}", _sum);
}
