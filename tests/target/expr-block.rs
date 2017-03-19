// rustfmt-array_layout: Block
// Test expressions with block formatting.

fn arrays() {
    [];
    let empty = [];

    let foo = [a_long_name, a_very_lng_name, a_long_name, a_very_lng_name, a_long_name];

    let foo = [
        a_long_name,
        a_very_lng_name,
        a_long_name,
        a_very_lng_name,
        a_long_name,
        a_very_lng_name,
        a_long_name,
        a_very_lng_name,
    ];

    vec![
        a_long_name,
        a_very_lng_name,
        a_long_name,
        a_very_lng_name,
        a_long_name,
        a_very_lng_name,
        a_very_lng_name,
    ];

    [
        a_long_name,
        a_very_lng_name,
        a_long_name,
        a_very_lng_name,
        a_long_name,
        a_very_lng_name,
        a_very_lng_name,
    ]
}

fn arrays() {
    let x = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        0,
        7,
        8,
        9,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        0,
    ];

    let y = [
        /* comment */
        1,
        2, /* post comment */
        3,
    ];

    let xy = [
        strukt {
            test123: value_one_two_three_four,
            turbo: coolio(),
        },
        /* comment  */
        1,
    ];

    let a = WeightedChoice::new(&mut [
        Weighted {
            weight: x,
            item: 0,
        },
        Weighted {
            weight: 1,
            item: 1,
        },
        Weighted {
            weight: x,
            item: 2,
        },
        Weighted {
            weight: 1,
            item: 3,
        },
    ]);

    let z =
        [xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx, yyyyyyyyyyyyyyyyyyyyyyyyyyy, zzzzzzzzzzzzzzzzz, q];

    [1 + 3, 4, 5, 6, 7, 7, fncall::<Vec<_>>(3 - 1)]
}
