use std::io::Write;
use std::fmt;

struct Foo<'a> {
    writer: &'a mut (Write+'a),
    other: &'a str,
}

struct Bar;

impl fmt::Write for Bar {
    fn write_str(&mut self, _: &str) -> fmt::Result {
        Ok(())
    }
}

fn borrowing_writer_from_struct_and_formatting_struct_field(foo: Foo) {
    write!(foo.writer, "{}", foo.other);
}

fn main() {
    let mut w = Vec::new();
    write!(&mut w as &mut Write, "");
    write!(&mut w, ""); // should coerce
    println!("ok");

    let mut s = Bar;
    {
        use std::fmt::Write;
        write!(&mut s, "test");
    }
}
