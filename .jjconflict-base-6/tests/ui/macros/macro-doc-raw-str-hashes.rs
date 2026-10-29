//@ run-pass
// The number of `#`s used to wrap the documentation comment should differ regarding the content.
//
// Related issue: #27489

macro_rules! homura {
    ($x:expr, #[$y:meta]) => (assert_eq!($x, stringify!($y)))
}

fn main() {
    homura! {
        r#"doc = r" Madoka""#,
        /// Madoka
    };

    homura! {
        r##"doc = r#" One quote mark: ["]"#"##,
        /// One quote mark: ["]
    };

    homura! {
        r##"doc = r#" Two quote marks: [""]"#"##,
        /// Two quote marks: [""]
    };

    homura! {
        r#####"doc = r####" Raw string ending sequences: ["###]"####"#####,
        /// Raw string ending sequences: ["###]
    };
}
