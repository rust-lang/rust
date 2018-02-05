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
