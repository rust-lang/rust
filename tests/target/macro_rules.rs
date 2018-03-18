// rustfmt-error_on_line_overflow: false

macro_rules! m {
    () => {};
    ($x: ident) => {};
    ($m1: ident, $m2: ident, $x: ident) => {};
    ($($beginning: ident),*; $middle: ident; $($end: ident),*) => {};
    (
        $($beginning: ident),*;
        $middle: ident;
        $($end: ident),*;
        $($beginning: ident),*;
        $middle: ident;
        $($end: ident),*
    ) => {};
    ($name: ident($($dol: tt $var: ident)*) $($body: tt)*) => {};
    (
        $($i: ident: $ty: ty, $def: expr, $stb: expr, $($dstring: tt),+);+ $(;)*
        $($i: ident: $ty: ty, $def: expr, $stb: expr, $($dstring: tt),+);+ $(;)*
    ) => {};
    ($foo: tt foo[$attr: meta] $name: ident) => {};
    ($foo: tt[$attr: meta] $name: ident) => {};
    ($foo: tt &'a[$attr: meta] $name: ident) => {};
    ($foo: tt foo #[$attr: meta] $name: ident) => {};
    ($foo: tt #[$attr: meta] $name: ident) => {};
    ($foo: tt &'a #[$attr: meta] $name: ident) => {};
    ($x: tt foo bar foo bar foo bar $y: tt => x * y * z $z: tt, $($a: tt),*) => {};
}

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

// #2438, #2476
macro_rules! m {
    () => {
        fn foo() {
            this_line_is_98_characters_long_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx();
        }
    };
}
macro_rules! m {
    () => {
        fn foo() {
            this_line_is_99_characters_long_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(
            );
        }
    };
}
macro_rules! m {
    () => {
        fn foo() {
            this_line_is_100_characters_long_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(
            );
        }
    };
}
macro_rules! m {
    () => {
        fn foo() {
            this_line_is_101_characters_long_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(
            );
        }
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

// #2470
macro foo($type_name: ident, $docs: expr) {
    #[allow(non_camel_case_types)]
    #[doc=$docs]
    #[derive(Debug, Clone, Copy)]
    pub struct $type_name;
}

// #2538
macro_rules! add_message_to_notes {
    ($msg: expr) => {{
        let mut lines = message.lines();
        notes.push_str(&format!("\n{}: {}", level, lines.next().unwrap()));
        for line in lines {
            notes.push_str(&format!(
                "\n{:indent$}{line}",
                "",
                indent = level.len() + 2,
                line = line,
            ));
        }
    }};
}
