#![deny(unused_must_use)]

fn error() -> Result<(), ()> {
    Err(())
}

macro_rules! foo {
    () => {{
        error();
    }};
}

fn main() {
    let _ = foo!(); //~ ERROR unused `Result` that must be used
}
