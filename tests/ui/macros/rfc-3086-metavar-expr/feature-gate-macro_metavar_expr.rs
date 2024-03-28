//@ run-pass

#![feature(macro_metavar_expr)]

/// Generate macros that count idents and then add a constant number
/// to the count.
///
/// This macro uses dollar escaping to make it unambiguous as to which
/// macro the repetition belongs to.
macro_rules! make_count_adders {
    ( $( $i:ident, $b:literal );* ) => {
        $(
            macro_rules! $i {
                ( $$( $$j:ident ),* ) => {
                    $b + $${count($j)}
                };
            }
        )*
    };
}

make_count_adders! { plus_one, 1; plus_five, 5 }

/// Generate a macro that allows selection of a particular literal
/// from a sequence of inputs by their identifier.
///
/// This macro uses dollar escaping to make it unambiguous as to which
/// macro the repetition belongs to, and to allow expansion of an
/// identifier the name of which is not known in the definition
/// of `make_picker`.
macro_rules! make_picker {
    ( $m:ident => $( $i:ident ),* ; $p:ident ) => {
        macro_rules! $m {
            ( $( $$ $i:literal ),* ) => {
                $$ $p
            };
        }
    };
}

make_picker!(first => a, b; a);

make_picker!(second => a, b; b);

fn main() {
    assert_eq!(plus_one!(a, b, c), 4);
    assert_eq!(plus_five!(a, b), 7);
    assert_eq!(first!(1, 2), 1);
    assert_eq!(second!(1, 2), 2);
}
