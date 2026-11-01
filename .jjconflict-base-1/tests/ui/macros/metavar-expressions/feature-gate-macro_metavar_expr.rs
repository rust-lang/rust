//@ run-pass

#![feature(macro_metavar_expr)]

/// Count the number of idents in a macro repetition.
macro_rules! count_idents {
    ( $( $i:ident ),* ) => {
        ${count($i)}
    };
}

/// Count the number of idents in a 2-dimensional macro repetition.
macro_rules! count_idents_2 {
    ( $( [ $( $i:ident ),* ] ),* ) => {
        ${count($i)}
    };
}

/// Mostly counts the number of OUTER-MOST repetitions
macro_rules! count_depth_limits {
    ( $( { $( [ $( $outer:ident : ( $( $inner:ident )* ) )* ] )* } )* ) => {
        (
            (
                ${count($inner)},
                ${count($inner, 0)},
                ${count($inner, 1)},
                ${count($inner, 2)},
                ${count($inner, 3)},
            ),
            (
                ${count($outer)},
                ${count($outer, 0)},
                ${count($outer, 1)},
                ${count($outer, 2)},
            ),
        )
    };
}

/// Produce (index, len) pairs for literals in a macro repetition.
/// The literal is not included in the output, so this macro uses the
/// `ignore` meta-variable expression to create a non-expanding
/// repetition binding.
macro_rules! enumerate_literals {
    ( $( ($l:stmt) ),* ) => {
        [$( ${ignore($l)} (${index()}, ${len()}) ),*]
    };
}

/// Produce index and len tuples for literals in a 2-dimensional
/// macro repetition.
macro_rules! enumerate_literals_2 {
    ( $( [ $( ($l:literal) ),* ] ),* ) => {
        [
            $(
                $(
                    (
                        ${index(1)},
                        ${len(1)},
                        ${index(0)},
                        ${len(0)},
                        $l
                    ),
                )*
            )*
        ]
    };
}

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
    assert_eq!(count_idents!(a, b, c), 3);
    assert_eq!(count_idents_2!([a, b, c], [d, e], [f]), 6);
    assert_eq!(
        count_depth_limits! {
            {
                [ A: (a b c) D: (d e f) ]
                [ G: (g h) I: (i j k l m) ]
                [ N: (n) ]
            }
            {
                [ O: (o) P: (p q) R: (r s) ]
                [ T: (t u v w x y z) ]
            }
        },
        ((26, 26, 9, 5, 2), (9, 9, 5, 2))
    );
    assert_eq!(enumerate_literals![("foo"), ("bar")], [(0, 2), (1, 2)]);
    assert_eq!(
        enumerate_literals_2![
            [("foo"), ("bar"), ("baz")],
            [("qux"), ("quux"), ("quuz"), ("xyzzy")]
        ],
        [
            (0, 2, 0, 3, "foo"),
            (0, 2, 1, 3, "bar"),
            (0, 2, 2, 3, "baz"),
            (1, 2, 0, 4, "qux"),
            (1, 2, 1, 4, "quux"),
            (1, 2, 2, 4, "quuz"),
            (1, 2, 3, 4, "xyzzy"),
        ]
    );
    assert_eq!(plus_one!(a, b, c), 4);
    assert_eq!(plus_five!(a, b), 7);
    assert_eq!(first!(1, 2), 1);
    assert_eq!(second!(1, 2), 2);
}
