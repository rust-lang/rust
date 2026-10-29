//@ run-pass

#![feature(macro_metavar_expr)]

fn main() {
    macro_rules! one_nested_count_and_len {
        ( $( [ $( $l:literal ),* ] ),* ) => {
            [
                // outermost repetition
                $(
                    // innermost repetition
                    $(
                        ${ignore($l)} ${index()}, ${len()},
                    )*
                    ${count($l)}, ${index()}, ${len()},
                )*
                ${count($l)},
            ]
        };
    }
    assert_eq!(
        one_nested_count_and_len!(["foo"], ["bar", "baz"]),
        [
            // # ["foo"]

            // ## innermost repetition (first iteration)
            //
            // `index` is 0 because this is the first innermost iteration.
            // `len` is 1 because there is only one innermost repetition, "foo".
            0, 1,
            // ## outermost repetition (first iteration)
            //
            // `count` is 1 because of "foo", i,e, `$l` has only one repetition,
            // `index` is 0 because this is the first outermost iteration.
            // `len` is 2 because there are 2 outermost repetitions, ["foo"] and ["bar", "baz"]
            1, 0, 2,
            // # ["bar", "baz"]

            // ## innermost repetition (first iteration)
            //
            // `index` is 0 because this is the first innermost iteration
            // `len` is 2 because there are repetitions, "bar" and "baz"
            0, 2,
            // ## innermost repetition (second iteration)
            //
            // `index` is 1 because this is the second innermost iteration
            // `len` is 2 because there are repetitions, "bar" and "baz"
            1, 2,
            // ## outermost repetition (second iteration)
            //
            // `count` is 2 because of "bar" and "baz", i,e, `$l` has two repetitions,
            // `index` is 1 because this is the second outermost iteration
            // `len` is 2 because there are 2 outermost repetitions, ["foo"] and ["bar", "baz"]
            2, 1, 2,
            // # last count

            // Because there are a total of 3 repetitions of `$l`, "foo", "bar" and "baz"
            3,
        ]
    );

    // Based on the above explanation, the following macros should be straightforward

    // Grouped from the outermost to the innermost
    macro_rules! three_nested_count {
        ( $( { $( [ $( ( $( $i:ident )* ) )* ] )* } )* ) => {
            &[
                $( $( $(
                    &[
                        ${ignore($i)} ${count($i, 0)},
                    ][..],
                )* )* )*

                $( $(
                    &[
                        ${ignore($i)} ${count($i, 0)},
                        ${ignore($i)} ${count($i, 1)},
                    ][..],
                )* )*

                $(
                    &[
                        ${ignore($i)} ${count($i, 0)},
                        ${ignore($i)} ${count($i, 1)},
                        ${ignore($i)} ${count($i, 2)},
                    ][..],
                )*

                &[
                    ${count($i, 0)},
                    ${count($i, 1)},
                    ${count($i, 2)},
                    ${count($i, 3)},
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
            &[6, 2][..],
            // (g h) (i j k l m)
            &[7, 2][..],
            // (n)
            &[1, 1][..],
            // (o) (p q) (r s)
            &[5, 3][..],
            // (t u v w x y z)
            &[7, 1][..],
            // [ (a b c) (d e f) ]
            // [ (g h) (i j k l m) ]
            // [ (n) ]
            &[14, 5, 3][..],
            // [ (o) (p q) (r s) ]
            // [ (t u v w x y z) ]
            &[12, 4, 2][..],
            // {
            //     [ (a b c) (d e f) ]
            //     [ (g h) (i j k l m) ]
            //     [ (n) ]
            // }
            // {
            //     [ (o) (p q) (r s) ]
            //     [ (t u v w x y z) ]
            // }
            &[26, 9, 5, 2][..]
        ][..]
    );

    // Grouped from the outermost to the innermost
    macro_rules! three_nested_len {
        ( $( { $( [ $( ( $( $i:ident )* ) )* ] )* } )* ) => {
            &[
                $( $( $( $(
                    &[
                        ${ignore($i)} ${len(3)},
                        ${ignore($i)} ${len(2)},
                        ${ignore($i)} ${len(1)},
                        ${ignore($i)} ${len(0)},
                    ][..],
                )* )* )* )*

                $( $( $(
                    &[
                        ${ignore($i)} ${len(2)},
                        ${ignore($i)} ${len(1)},
                        ${ignore($i)} ${len(0)},
                    ][..],
                )* )* )*

                $( $(
                    &[
                        ${ignore($i)} ${len(1)},
                        ${ignore($i)} ${len(0)},
                    ][..],
                )* )*

                $(
                    &[
                        ${ignore($i)} ${len(0)},
                    ][..],
                )*
            ][..]
        }
    }
    assert_eq!(
        three_nested_len!(
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
            &[2, 3, 2, 3][..],
            &[2, 3, 2, 3][..],
            &[2, 3, 2, 3][..],
            // d e f
            &[2, 3, 2, 3][..],
            &[2, 3, 2, 3][..],
            &[2, 3, 2, 3][..],
            // g h
            &[2, 3, 2, 2][..],
            &[2, 3, 2, 2][..],
            // i j k l m
            &[2, 3, 2, 5][..],
            &[2, 3, 2, 5][..],
            &[2, 3, 2, 5][..],
            &[2, 3, 2, 5][..],
            &[2, 3, 2, 5][..],
            // n
            &[2, 3, 1, 1][..],
            // o
            &[2, 2, 3, 1][..],
            // p q
            &[2, 2, 3, 2][..],
            &[2, 2, 3, 2][..],
            // r s
            &[2, 2, 3, 2][..],
            &[2, 2, 3, 2][..],
            // t u v w x y z
            &[2, 2, 1, 7][..],
            &[2, 2, 1, 7][..],
            &[2, 2, 1, 7][..],
            &[2, 2, 1, 7][..],
            &[2, 2, 1, 7][..],
            &[2, 2, 1, 7][..],
            &[2, 2, 1, 7][..],
            // (a b c) (d e f)
            &[2, 3, 2][..],
            &[2, 3, 2][..],
            // (g h) (i j k l m)
            &[2, 3, 2][..],
            &[2, 3, 2][..],
            // (n)
            &[2, 3, 1][..],
            // (o) (p q) (r s)
            &[2, 2, 3][..],
            &[2, 2, 3][..],
            &[2, 2, 3][..],
            // (t u v w x y z)
            &[2, 2, 1][..],
            // [ (a b c) (d e f) ]
            // [ (g h) (i j k l m) ]
            // [ (n) ]
            &[2, 3][..],
            &[2, 3][..],
            &[2, 3,][..],
            // [ (o) (p q) (r s) ]
            // [ (t u v w x y z) ]
            &[2, 2][..],
            &[2, 2][..],
            // {
            //     [ (a b c) (d e f) ]
            //     [ (g h) (i j k l m) ]
            //     [ (n) ]
            // }
            // {
            //     [ (o) (p q) (r s) ]
            //     [ (t u v w x y z) ]
            // }
            &[2][..],
            &[2][..]
        ][..]
    );

    // It is possible to say, to some degree, that count is an "amalgamation" of len (see
    // each len line result and compare them with the count results)
}
