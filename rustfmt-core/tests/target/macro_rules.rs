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

// #2439
macro_rules! m {
    (
        $line0_xxxxxxxxxxxxxxxxx: expr,
        $line1_xxxxxxxxxxxxxxxxx: expr,
        $line2_xxxxxxxxxxxxxxxxx: expr,
        $line3_xxxxxxxxxxxxxxxxx: expr,
    ) => {};
}
