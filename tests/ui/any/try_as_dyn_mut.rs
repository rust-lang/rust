//@ run-pass
#![feature(try_as_dyn)]

use std::fmt::{Error, Write};

// Look ma, no `T: Write`
fn try_as_dyn_mut_write<T: 'static>(t: &mut T, s: &str) -> Result<(), Error> {
    match std::any::try_as_dyn_mut::<_, dyn Write>(t) {
        Some(w) => w.write_str(s),
        None => Ok(())
    }
}

// Test that downcasting to a mut dyn trait works as expected
fn main() {
    let mut buf = "Hello".to_string();

    try_as_dyn_mut_write(&mut buf, " world!").unwrap();
    assert_eq!(buf, "Hello world!");
}
