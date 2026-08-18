//@no-rustfix
#![warn(clippy::write_literal)]

use std::io::Write;

fn escaping() {
    let mut v = vec![];

    writeln!(v, r"{}", '"');
    //~^ write_literal

    // hard mode
    writeln!(v, r#"{}{}"#, '#', '"');
    //~^ write_literal
}
