//@ run-pass

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

/// Produce (index, length) pairs for literals in a macro repetition.
/// The literal is not included in the output, so this macro uses the
/// `ignore` meta-variable expression to create a non-expanding
/// repetition binding.
macro_rules! enumerate_literals {
    ( $( ($l:stmt) ),* ) => {
        [$( ${ignore($l)} (${index()}, ${length()}) ),*]
    };
}

/// Produce index and length tuples for literals in a 2-dimensional
/// macro repetition.
macro_rules! enumerate_literals_2 {
    ( $( [ $( ($l:literal) ),* ] ),* ) => {
        [
            $(
                $(
                    (
                        ${index(1)},
                        ${length(1)},
                        ${index(0)},
                        ${length(0)},
                        $l
                    ),
                )*
            )*
        ]
    };
}

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
}
