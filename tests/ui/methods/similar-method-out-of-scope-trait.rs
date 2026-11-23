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

// A method from the missing trait must not hide a similarly named method that can be called.
mod hidden {
    pub struct ByValue;
    pub struct Ambiguous;
    pub struct Inherent;

    pub trait Hidden {
        fn method(&self) {}
        fn method2(self)
        where
            Self: Sized,
        {
        }
    }

    impl Hidden for ByValue {}
    impl Hidden for Inherent {}

    pub trait HiddenRef {
        fn method(&self) {}
        fn method2(&self) {}
    }

    impl HiddenRef for Ambiguous {}

    impl Inherent {
        pub fn method2(&self) {}
    }
}

trait Visible {
    fn method2(&self) {}
}

impl Visible for hidden::ByValue {}
impl Visible for hidden::Ambiguous {}

fn in_scope_candidates() {
    hidden::ByValue.method();
    //~^ ERROR no method named `method` found
    hidden::Ambiguous.method();
    //~^ ERROR no method named `method` found
    hidden::Inherent.method();
    //~^ ERROR no method named `method` found
}
