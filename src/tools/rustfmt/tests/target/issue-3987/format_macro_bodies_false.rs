// rustfmt-format_macro_bodies: false

// with comments
macro_rules! macros {
    () => {{
        Struct {
            field: (
                42 + //comment 1
                42
                //comment 2
            ),
        };
    }};
}

// without comments
macro_rules! macros {
    () => {{
        Struct {
            field: (
                42 +
                42
            ),
        };
    }};
}
