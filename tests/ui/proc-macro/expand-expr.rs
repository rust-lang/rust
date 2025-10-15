//@ proc-macro: expand-expr.rs
//@ ignore-backends: gcc
// No `remap-src-base`, since `check_expand_expr_file!()` fails when enabled.

#![feature(concat_bytes)]
extern crate expand_expr;

use expand_expr::{
    check_expand_expr_file, echo_pm, expand_expr_fail, expand_expr_is, recursive_expand,
};

// Check builtin macros can be expanded.

expand_expr_is!(14u32, line!());
expand_expr_is!(24u32, column!());

expand_expr_is!("Hello, World!", concat!("Hello, ", "World", "!"));
expand_expr_is!("int10floats5.3booltrue", concat!("int", 10, "floats", 5.3, "bool", true));
expand_expr_is!("Hello", concat!(r##"Hello"##));

expand_expr_is!("Included file contents\n", include_str!("auxiliary/included-file.txt"));
expand_expr_is!(b"Included file contents\n", include_bytes!("auxiliary/included-file.txt"));

expand_expr_is!(
    "contents: Included file contents\n",
    concat!("contents: ", include_str!("auxiliary/included-file.txt"))
);

expand_expr_is!(
    b"contents: Included file contents\n",
    concat_bytes!(b"contents: ", include_bytes!("auxiliary/included-file.txt"))
);

// Correct value is checked for multiple sources.
check_expand_expr_file!(file!());

expand_expr_is!("hello", stringify!(hello));
expand_expr_is!("10 + 20", stringify!(10 + 20));

macro_rules! echo_tts {
    ($($t:tt)*) => { $($t)* };
}

macro_rules! echo_lit {
    ($l:literal) => {
        $l
    };
}

macro_rules! echo_expr {
    ($e:expr) => {
        $e
    };
}

macro_rules! simple_lit {
    ($l:literal) => {
        expand_expr_is!($l, $l);
        expand_expr_is!($l, echo_lit!($l));
        expand_expr_is!($l, echo_expr!($l));
        expand_expr_is!($l, echo_tts!($l));
        expand_expr_is!($l, echo_pm!($l));
        const _: () = {
            macro_rules! mac {
                () => {
                    $l
                };
            }
            expand_expr_is!($l, mac!());
            expand_expr_is!($l, echo_expr!(mac!()));
            expand_expr_is!($l, echo_tts!(mac!()));
            expand_expr_is!($l, echo_pm!(mac!()));
        };
    };
}

simple_lit!("Hello, World");
simple_lit!('c');
simple_lit!(b'c');
simple_lit!(10);
simple_lit!(10.0);
simple_lit!(10.0f64);
simple_lit!(-3.14159);
simple_lit!(-3.5e10);
simple_lit!(0xFEED);
simple_lit!(-0xFEED);
simple_lit!(0b0100);
simple_lit!(-0b0100);
simple_lit!("string");
simple_lit!(r##"raw string"##);
simple_lit!(b"byte string");
simple_lit!(br##"raw byte string"##);
simple_lit!(true);
simple_lit!(false);

// Ensure char escapes aren't normalized by expansion
simple_lit!("\u{0}");
simple_lit!("\0");
simple_lit!("\x00");
simple_lit!('\u{0}');
simple_lit!('\0');
simple_lit!('\x00');
simple_lit!(b"\x00");
simple_lit!(b"\0");
simple_lit!(b'\x00');
simple_lit!(b'\0');

// Extra tokens after the string literal aren't ignored
expand_expr_fail!("string"; hello); //~ ERROR: expected one of `.`, `?`, or an operator, found `;`

// Invalid expressions produce errors in addition to returning `Err(())`.
expand_expr_fail!($); //~ ERROR: expected expression, found `$`
expand_expr_fail!(echo_tts!($)); //~ ERROR: expected expression, found `$`
expand_expr_fail!(echo_pm!($)); //~ ERROR: expected expression, found `$`

// We get errors reported and recover during macro expansion if the macro
// doesn't produce a valid expression.
expand_expr_is!("string", echo_tts!("string"; hello)); //~ ERROR: macro expansion ignores `hello` and any tokens following
expand_expr_is!("string", echo_pm!("string"; hello)); //~ ERROR: macro expansion ignores `;` and any tokens following

// For now, fail if a non-literal expression is expanded.
expand_expr_fail!(arbitrary_expression() + "etc");
expand_expr_fail!(echo_tts!(arbitrary_expression() + "etc"));
expand_expr_fail!(echo_expr!(arbitrary_expression() + "etc"));
expand_expr_fail!(echo_pm!(arbitrary_expression() + "etc"));

const _: u32 = recursive_expand!(); //~ ERROR: recursion limit reached while expanding `recursive_expand!`

fn main() {
    // https://github.com/rust-lang/rust/issues/104414
    match b"Included file contents\n" {
        include_bytes!("auxiliary/included-file.txt") => (),
        _ => panic!("include_bytes! in pattern"),
    }
}
