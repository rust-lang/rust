#![warn(clippy::literal_string_with_formatting_arg)]
#![allow(clippy::unnecessary_literal_unwrap)]

fn main() {
    let x: Option<usize> = None;
    let y = "hello";
    x.expect("{y} {}"); //~ literal_string_with_formatting_arg
    x.expect("{:?}"); //~ literal_string_with_formatting_arg
    x.expect("{y:?}"); //~ literal_string_with_formatting_arg
    x.expect(" {y:?} {y:?} "); //~ literal_string_with_formatting_arg
    x.expect(" {y:..} {y:?} "); //~ literal_string_with_formatting_arg
    x.expect(r"{y:?}  {y:?} "); //~ literal_string_with_formatting_arg
    x.expect(r"{y:?} y:?}"); //~ literal_string_with_formatting_arg
    x.expect(r##" {y:?} {y:?} "##); //~ literal_string_with_formatting_arg
    "\\.+*?()|[]{}^$#&-~".chars().any(|x| x == 'a'); //~ literal_string_with_formatting_arg
    // Ensure that it doesn't try to go in the middle of a unicode character.
    x.expect("———{}"); //~ literal_string_with_formatting_arg

    // Should not lint!
    format!("{y:?}");
    println!("{y:?}");
    x.expect("{{y} {x");
    x.expect("{{y:?}");
    x.expect("{y:...}");
    let _ = "fn main {\n\
    }";
    // Unicode characters escape should not lint either.
    "\u{0052}".to_string();
}
