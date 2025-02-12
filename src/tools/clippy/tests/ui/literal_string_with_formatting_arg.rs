#![warn(clippy::literal_string_with_formatting_args)]
#![allow(clippy::unnecessary_literal_unwrap)]

fn main() {
    let x: Option<usize> = None;
    let y = "hello";
    x.expect("{y} {}"); //~ literal_string_with_formatting_args
    x.expect(" {y} bla"); //~ literal_string_with_formatting_args
    x.expect("{:?}"); //~ literal_string_with_formatting_args
    x.expect("{y:?}"); //~ literal_string_with_formatting_args
    x.expect(" {y:?} {y:?} "); //~ literal_string_with_formatting_args
    x.expect(" {y:..} {y:?} "); //~ literal_string_with_formatting_args
    x.expect(r"{y:?}  {y:?} "); //~ literal_string_with_formatting_args
    x.expect(r"{y:?} y:?}"); //~ literal_string_with_formatting_args
    x.expect(r##" {y:?} {y:?} "##); //~ literal_string_with_formatting_args
    // Ensure that it doesn't try to go in the middle of a unicode character.
    x.expect("———{:?}"); //~ literal_string_with_formatting_args

    // Should not lint!
    format!("{y:?}");
    println!("{y:?}");
    x.expect(" {} "); // We ignore `{}` to limit false positives.
    x.expect(" {   } "); // We ignore `{}` to limit false positives.
    x.expect("{{y} {x");
    x.expect("{{y:?}");
    x.expect(" {0}"); // If it only contains an integer, we ignore it.
    x.expect(r##" {x:?} "##); // `x` doesn't exist so we shoud not lint
    x.expect("{y:...}");
    let _ = "fn main {\n\
    }";
    // Unicode characters escape should not lint either.
    "\u{0052}".to_string();

    // Regression test for <https://github.com/rust-lang/rust-clippy/issues/13838>.
    let x: Option<usize> = Some(0);
    x.expect("{…}");
}
