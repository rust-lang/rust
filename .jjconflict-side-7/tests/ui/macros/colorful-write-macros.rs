//@ run-pass
#![allow(dead_code)]
use std::io::Write;
use std::fmt;

struct Foo<'a> {
    writer: &'a mut (dyn Write+'a),
    other: &'a str,
}

struct Bar;

impl fmt::Write for Bar {
    fn write_str(&mut self, _: &str) -> fmt::Result {
        Ok(())
    }
}

fn borrowing_writer_from_struct_and_formatting_struct_field(foo: Foo) {
    write!(foo.writer, "{}", foo.other).unwrap();
}

fn main() {
    let mut w = Vec::new();
    write!(&mut w as &mut dyn Write, "").unwrap();
    write!(&mut w, "").unwrap(); // should coerce
    println!("ok");

    let mut s = Bar;
    {
        use std::fmt::Write;
        write!(&mut s, "test").unwrap();
    }
}
