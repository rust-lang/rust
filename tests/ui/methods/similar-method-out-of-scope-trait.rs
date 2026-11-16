// Regression test for #165389: don't suggest another method from the same missing trait.

mod writer {
    pub struct Writer;

    pub trait Write {
        fn write(&self) {}
        fn write_all(&self) {}
    }

    impl Write for Writer {}

    impl Writer {
        pub fn flush(&self) {}
    }
}

trait Read {
    fn read(&self) {}
}

impl Read for writer::Writer {}

fn generic<T: writer::Write>(writer: T) {
    writer.writ();
    //~^ ERROR no method named `writ` found
}

fn trait_object(writer: &dyn writer::Write) {
    writer.writ();
    //~^ ERROR no method named `writ` found
}

fn main() {
    let writer = writer::Writer;
    writer.write_all();
    //~^ ERROR no method named `write_all` found
    writer.reed();
    //~^ ERROR no method named `reed` found
    writer.flus();
    //~^ ERROR no method named `flus` found
}
