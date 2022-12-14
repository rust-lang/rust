// run-pass

#![feature(macro_metavar_expr)]

fn main() {
    macro_rules! one_nested_count_and_length {
        ( $( [ $( $l:literal ),* ] ),* ) => {
            [
                // outer-most repetition
                $(
                    // inner-most repetition
                    $(
                        ${ignore(l)} ${index()}, ${length()},
                    )*
                    ${count(l)}, ${index()}, ${length()},
                )*
                ${count(l)},
            ]
        };
    }
    assert_eq!(
        one_nested_count_and_length!(["foo"], ["bar", "baz"]),
        [
            // # ["foo"]

            // ## inner-most repetition (first iteration)
            //
            // `index` is 0 because this is the first inner-most iteration.
            // `length` is 1 because there is only one inner-most repetition, "foo".
            0, 1,

            // ## outer-most repetition (first iteration)
            //
            // `count` is 1 because of "foo", i,e, `$l` has only one repetition,
            // `index` is 0 because this is the first outer-most iteration.
            // `length` is 2 because there are 2 outer-most repetitions, ["foo"] and ["bar", "baz"]
            1, 0, 2,

            // # ["bar", "baz"]

            // ## inner-most repetition (first iteration)
            //
            // `index` is 0 because this is the first inner-most iteration
            // `length` is 2 because there are repetitions, "bar" and "baz"
            0, 2,

            // ## inner-most repetition (second iteration)
            //
            // `index` is 1 because this is the second inner-most iteration
            // `length` is 2 because there are repetitions, "bar" and "baz"
            1, 2,

            // ## outer-most repetition (second iteration)
            //
            // `count` is 2 because of "bar" and "baz", i,e, `$l` has two repetitions,
            // `index` is 1 because this is the second outer-most iteration
            // `length` is 2 because there are 2 outer-most repetitions, ["foo"] and ["bar", "baz"]
            2, 1, 2,

            // # last count

            // Because there are a total of 3 repetitions of `$l`, "foo", "bar" and "baz"
            3,
        ]
    );

    // Based on the above explanation, the following macros should be straightforward

    // Grouped from the outer-most to the inner-most
    macro_rules! three_nested_count {
        ( $( { $( [ $( ( $( $i:ident )* ) )* ] )* } )* ) => {
            &[
                $( $( $(
                    &[
                        ${ignore(i)} ${count(i, 0)},
                    ][..],
                )* )* )*

                $( $(
                    &[
                        ${ignore(i)} ${count(i, 0)},
                        ${ignore(i)} ${count(i, 1)},
                    ][..],
                )* )*

                $(
                    &[
                        ${ignore(i)} ${count(i, 0)},
                        ${ignore(i)} ${count(i, 1)},
                        ${ignore(i)} ${count(i, 2)},
                    ][..],
                )*

                &[
                    ${count(i, 0)},
                    ${count(i, 1)},
                    ${count(i, 2)},
                    ${count(i, 3)},
                ][..]
            ][..]
        }
    }
    assert_eq!(
        three_nested_count!(
            {
                [ (a b c) (d e f) ]
                [ (g h) (i j k l m) ]
                [ (n) ]
            }
            {
                [ (o) (p q) (r s) ]
                [ (t u v w x y z) ]
            }
        ),
        &[
            // a b c
            &[3][..],
            // d e f
            &[3][..],
            // g h
            &[2][..],
            // i j k l m
            &[5][..],
            // n
            &[1][..],
            // o
            &[1][..],
            // p q
            &[2][..],
            // r s
            &[2][..],
            // t u v w x y z
            &[7][..],

            // (a b c) (d e f)
            &[2, 6][..],
            // (g h) (i j k l m)
            &[2, 7][..],
            // (n)
            &[1, 1][..],
            // (o) (p q) (r s)
            &[3, 5][..],
            // (t u v w x y z)
            &[1, 7][..],

            // [ (a b c) (d e f) ]
            // [ (g h) (i j k l m) ]
            // [ (n) ]
            &[3, 5, 14][..],
            // [ (o) (p q) (r s) ]
            // [ (t u v w x y z) ]
            &[2, 4, 12][..],

            // {
            //     [ (a b c) (d e f) ]
            //     [ (g h) (i j k l m) ]
            //     [ (n) ]
            // }
            // {
            //     [ (o) (p q) (r s) ]
            //     [ (t u v w x y z) ]
            // }
            &[2, 5, 9, 26][..]
        ][..]
    );

    // Grouped from the outer-most to the inner-most
    macro_rules! three_nested_length {
        ( $( { $( [ $( ( $( $i:ident )* ) )* ] )* } )* ) => {
            &[
                $( $( $( $(
                    &[
                        ${ignore(i)} ${length(3)},
                        ${ignore(i)} ${length(2)},
                        ${ignore(i)} ${length(1)},
                        ${ignore(i)} ${length(0)},
                    ][..],
                )* )* )* )*

                $( $( $(
                    &[
                        ${ignore(i)} ${length(2)},
                        ${ignore(i)} ${length(1)},
                        ${ignore(i)} ${length(0)},
                    ][..],
                )* )* )*

                $( $(
                    &[
                        ${ignore(i)} ${length(1)},
                        ${ignore(i)} ${length(0)},
                    ][..],
                )* )*

                $(
                    &[
                        ${ignore(i)} ${length(0)},
                    ][..],
                )*
            ][..]
        }
    }
    assert_eq!(
        three_nested_length!(
            {
                [ (a b c) (d e f) ]
                [ (g h) (i j k l m) ]
                [ (n) ]
            }
            {
                [ (o) (p q) (r s) ]
                [ (t u v w x y z) ]
            }
        ),
        &[
            // a b c
            &[2, 3, 2, 3][..], &[2, 3, 2, 3][..], &[2, 3, 2, 3][..],
            // d e f
            &[2, 3, 2, 3][..], &[2, 3, 2, 3][..], &[2, 3, 2, 3][..],
            // g h
            &[2, 3, 2, 2][..], &[2, 3, 2, 2][..],
            // i j k l m
            &[2, 3, 2, 5][..], &[2, 3, 2, 5][..], &[2, 3, 2, 5][..], &[2, 3, 2, 5][..],
            &[2, 3, 2, 5][..],
            // n
            &[2, 3, 1, 1][..],
            // o
            &[2, 2, 3, 1][..],
            // p q
            &[2, 2, 3, 2][..], &[2, 2, 3, 2][..],
            // r s
            &[2, 2, 3, 2][..], &[2, 2, 3, 2][..],
            // t u v w x y z
            &[2, 2, 1, 7][..], &[2, 2, 1, 7][..], &[2, 2, 1, 7][..], &[2, 2, 1, 7][..],
            &[2, 2, 1, 7][..], &[2, 2, 1, 7][..], &[2, 2, 1, 7][..],

            // (a b c) (d e f)
            &[2, 3, 2][..], &[2, 3, 2][..],
            // (g h) (i j k l m)
            &[2, 3, 2][..], &[2, 3, 2][..],
            // (n)
            &[2, 3, 1][..],
            // (o) (p q) (r s)
            &[2, 2, 3][..], &[2, 2, 3][..], &[2, 2, 3][..],
            // (t u v w x y z)
            &[2, 2, 1][..],

            // [ (a b c) (d e f) ]
            // [ (g h) (i j k l m) ]
            // [ (n) ]
            &[2, 3][..], &[2, 3][..],  &[2, 3,][..],
            // [ (o) (p q) (r s) ]
            // [ (t u v w x y z) ]
            &[2, 2][..], &[2, 2][..],

            // {
            //     [ (a b c) (d e f) ]
            //     [ (g h) (i j k l m) ]
            //     [ (n) ]
            // }
            // {
            //     [ (o) (p q) (r s) ]
            //     [ (t u v w x y z) ]
            // }
            &[2][..], &[2][..]
        ][..]
    );

    // It is possible to say, to some degree, that count is an "amalgamation" of length (see
    // each length line result and compare them with the count results)
}
