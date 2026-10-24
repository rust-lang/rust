//@ check-fail
//@ edition: 2018

#![feature(try_blocks)]
#![crate_type = "lib"]

// fine because the `;` discards the value
fn foo(a: &str, b: &str) -> i32 {
    try {
        let foo = std::fs::read_to_string(a)?;
        std::fs::write(b, foo);
    };
    4 + 10
}

// parses without the semicolon, but gives a type error
fn bar(a: &str, b: &str) -> i32 {
    try {
        let foo = std::fs::read_to_string(a)?;
        //~^ ERROR mismatched types
        std::fs::write(b, foo);
    }
    4 + 10
}
