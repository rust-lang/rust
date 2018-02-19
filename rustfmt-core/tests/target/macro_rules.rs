macro_rules! m {
    // a
    ($expr: expr, $($func: ident)*) => {{
        let x = $expr;
        $func(x)
    }};

    /* b */
    () => {
        /* c */
    };

    (@tag) => {};

    // d
    ($item: ident) => {
        mod macro_item {
            struct $item;
        }
    };
}

macro m2 {
    // a
    ($expr: expr, $($func: ident)*) => {{
        let x = $expr;
        $func(x)
    }}

    /* b */
    () => {
        /* c */
    }

    (@tag) => {}

    // d
    ($item: ident) => {
        mod macro_item {
            struct $item;
        }
    }
}

// #2438
macro_rules! m {
    () => {
        this_line_is_99_characters_long_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(
        ); // this line is drifting
    };
}

// #2439
macro_rules! m {
    (
        $line0_xxxxxxxxxxxxxxxxx: expr,
        $line1_xxxxxxxxxxxxxxxxx: expr,
        $line2_xxxxxxxxxxxxxxxxx: expr,
        $line3_xxxxxxxxxxxxxxxxx: expr,
    ) => {};
}

// #2466
// Skip formatting `macro_rules!` that are not using `{}`.
macro_rules! m (
    () => ()
);
macro_rules! m [
    () => ()
];
