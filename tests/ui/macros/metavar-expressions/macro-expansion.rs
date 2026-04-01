//@ run-pass

#![feature(macro_metavar_expr)]

#[derive(Debug)]
struct Example<'a> {
    _indexes: &'a [(u32, u32)],
    _counts: &'a [u32],
    _nested: Vec<Example<'a>>,
}

macro_rules! example {
    ( $( [ $( ( $( $x:ident )* ) )* ] )* ) => {
        Example {
            _indexes: &[],
            _counts: &[${count($x, 0)}, ${count($x, 1)}, ${count($x, 2)}],
            _nested: vec![
            $(
                Example {
                    _indexes: &[(${index()}, ${len()})],
                    _counts: &[${count($x, 0)}, ${count($x, 1)}],
                    _nested: vec![
                    $(
                        Example {
                            _indexes: &[(${index(1)}, ${len(1)}), (${index()}, ${len()})],
                            _counts: &[${count($x)}],
                            _nested: vec![
                            $(
                                Example {
                                    _indexes: &[
                                        (${index(2)}, ${len(2)}),
                                        (${index(1)}, ${len(1)}),
                                        (${index()}, ${len()})
                                    ],
                                    _counts: &[],
                                    _nested: vec![],
                                    ${ignore($x)}
                                }
                            ),*
                            ]
                        }
                    ),*
                    ]
                }
            ),*
            ]
        }
    };
}

static EXPECTED: &str = concat!(
    "Example { _indexes: [], _counts: [13, 4, 2], _nested: [",
    concat!(
        "Example { _indexes: [(0, 2)], _counts: [10, 3], _nested: [",
        concat!(
            "Example { _indexes: [(0, 2), (0, 3)], _counts: [4], _nested: [",
            concat!(
                "Example { _indexes: [(0, 2), (0, 3), (0, 4)], _counts: [], _nested: [] }, ",
                "Example { _indexes: [(0, 2), (0, 3), (1, 4)], _counts: [], _nested: [] }, ",
                "Example { _indexes: [(0, 2), (0, 3), (2, 4)], _counts: [], _nested: [] }, ",
                "Example { _indexes: [(0, 2), (0, 3), (3, 4)], _counts: [], _nested: [] }",
            ),
            "] }, ",
            "Example { _indexes: [(0, 2), (1, 3)], _counts: [4], _nested: [",
            concat!(
                "Example { _indexes: [(0, 2), (1, 3), (0, 4)], _counts: [], _nested: [] }, ",
                "Example { _indexes: [(0, 2), (1, 3), (1, 4)], _counts: [], _nested: [] }, ",
                "Example { _indexes: [(0, 2), (1, 3), (2, 4)], _counts: [], _nested: [] }, ",
                "Example { _indexes: [(0, 2), (1, 3), (3, 4)], _counts: [], _nested: [] }",
            ),
            "] }, ",
            "Example { _indexes: [(0, 2), (2, 3)], _counts: [2], _nested: [",
            concat!(
                "Example { _indexes: [(0, 2), (2, 3), (0, 2)], _counts: [], _nested: [] }, ",
                "Example { _indexes: [(0, 2), (2, 3), (1, 2)], _counts: [], _nested: [] }",
            ),
            "] }",
        ),
        "] }, ",
        "Example { _indexes: [(1, 2)], _counts: [3, 1], _nested: [",
        concat!(
            "Example { _indexes: [(1, 2), (0, 1)], _counts: [3], _nested: [",
            concat!(
                "Example { _indexes: [(1, 2), (0, 1), (0, 3)], _counts: [], _nested: [] }, ",
                "Example { _indexes: [(1, 2), (0, 1), (1, 3)], _counts: [], _nested: [] }, ",
                "Example { _indexes: [(1, 2), (0, 1), (2, 3)], _counts: [], _nested: [] }",
            ),
            "] }",
        ),
        "] }",
    ),
    "] }",
);

fn main() {
    let e = example! {
        [ ( A B C D ) ( E F G H ) ( I J ) ]
        [ ( K L M ) ]
    };
    let debug = format!("{:?}", e);
    assert_eq!(debug, EXPECTED);
}
