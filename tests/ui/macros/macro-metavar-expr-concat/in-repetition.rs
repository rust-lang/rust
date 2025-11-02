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
                ${concat(_, $arg)} //~ ERROR nested repetitions with `${concat(...)}` metavariable expressions are not yet supported
            )*
        )*
    };
}
InRepetition!(other);

fn main() {}
