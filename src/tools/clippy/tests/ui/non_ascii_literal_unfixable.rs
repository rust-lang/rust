//@no-rustfix
#![warn(clippy::non_ascii_literal)]
#![allow(dead_code)]

fn non_ascii() {
    print!(r"€");
    //~^ non_ascii_literal
    print!(r"Üben!");
    //~^ non_ascii_literal
    print!(r"an en dash –");
    //~^ non_ascii_literal
    print!(r#"€ between hashes"#);
    //~^ non_ascii_literal
}

fn main() {}
