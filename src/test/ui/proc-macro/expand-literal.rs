// aux-build:expand-literal.rs

extern crate expand_literal;

use expand_literal::{echo_pm, expand_literal_is, recursive_expand};

// Check builtin macros can be expanded.

expand_literal_is!(9u32, line!());
expand_literal_is!(27u32, column!());

expand_literal_is!("Hello, World!", concat!("Hello, ", "World", "!"));
expand_literal_is!("int10floats5.3booltrue", concat!("int", 10, "floats", 5.3, "bool", true));
expand_literal_is!("Hello", concat!(r##"Hello"##));

expand_literal_is!("Included file contents\n", include_str!("auxiliary/included-file.txt"));
expand_literal_is!(b"Included file contents\n", include_bytes!("auxiliary/included-file.txt"));

expand_literal_is!(
    "contents: Included file contents\n",
    concat!("contents: ", include_str!("auxiliary/included-file.txt"))
);

// Correct value is checked using stderr checking to handle paths
expand_literal_is!("", file!()); //~ ERROR: proc macro panicked

expand_literal_is!("hello", stringify!(hello));
expand_literal_is!("10 + 20", stringify!(10 + 20));

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
        simple_lit!($l, $l);
    };
    ($l:literal, $e:literal) => {
        expand_literal_is!($e, $l);
        expand_literal_is!($e, echo_lit!($l));
        expand_literal_is!($e, echo_expr!($l));
        expand_literal_is!($e, echo_tts!($l));
        expand_literal_is!($e, echo_pm!($l));
        const _: () = {
            macro_rules! mac {
                () => {
                    $l
                };
            }
            expand_literal_is!($e, mac!());
            expand_literal_is!($e, echo_expr!(mac!()));
            expand_literal_is!($e, echo_tts!(mac!()));
            expand_literal_is!($e, echo_pm!(mac!()));
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
simple_lit!("string");
simple_lit!(r##"raw string"##);
simple_lit!(b"byte string");

// After expansion, all byte strings are considered non-raw.
simple_lit!(br##"raw byte string"##, b"raw byte string");

// After expansion, char escapes are normalized
simple_lit!("\u{0}");
simple_lit!("\0", "\u{0}");
simple_lit!("\x00", "\u{0}");

simple_lit!('\u{0}');
simple_lit!('\0', '\u{0}');
simple_lit!('\x00', '\u{0}');

simple_lit!(b"\x00");
simple_lit!(b"\0", b"\x00");

simple_lit!(b'\x00');
simple_lit!(b'\0', b'\x00');

// After expansion, all integer literals are decimal
simple_lit!(0xFEED, 65261);
simple_lit!(-0xFEED, -65261);
simple_lit!(0b0100, 4);
simple_lit!(-0b0100, -4);

// Booleans aren't considered literals by `expand_literal`
expand_literal_is!("fail", true); //~ ERROR: proc macro panicked
expand_literal_is!("fail", false); //~ ERROR: proc macro panicked
expand_literal_is!("fail", echo_tts!(true)); //~ ERROR: proc macro panicked
expand_literal_is!("fail", echo_tts!(false)); //~ ERROR: proc macro panicked

// Extra tokens after the string literal aren't ignored
expand_literal_is!("string", "string"; hello); //~ ERROR: proc macro panicked

// FIXME: We get errors reported and recover during macro expansion if the macro
// doesn't produce a valid expression.
expand_literal_is!("string", echo_tts!("string"; hello)); //~ ERROR: macro expansion ignores token `hello` and any following
expand_literal_is!("string", echo_pm!("string"; hello)); //~ ERROR: macro expansion ignores token `;` and any following

expand_literal_is!("fail", arbitrary_expression() + "etc"); //~ ERROR: proc macro panicked
expand_literal_is!("fail", echo_tts!(arbitrary_expression() + "etc")); //~ ERROR: proc macro panicked
expand_literal_is!("fail", echo_expr!(arbitrary_expression() + "etc")); //~ ERROR: proc macro panicked
expand_literal_is!("fail", echo_pm!(arbitrary_expression() + "etc")); //~ ERROR: proc macro panicked

const _: u32 = recursive_expand!(); //~ ERROR: recursion limit reached while expanding `recursive_expand!`

fn main() {}
