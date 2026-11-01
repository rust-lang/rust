// issue: <https://github.com/rust-lang/rust/issues/140479>
// Ensure a proper compiler error, instead of an ICE occurs.
// FIXME(macro_metavar_expr_concat): this error message could be improved
#![feature(macro_metavar_expr_concat)]

macro_rules! InRepetition {
    (
        $(
            $($arg:ident),+
        )+
     ) => {
        $(
            $(
                ${concat(_, $arg)} //~ ERROR macro expansion ends with an incomplete expression: expected one of `!` or `::`
            )*
        )*
    };
}
InRepetition!(other);

fn main() {}
