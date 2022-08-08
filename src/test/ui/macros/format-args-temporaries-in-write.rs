// check-pass
#![crate_type = "lib"]

use std::fmt::{self, Display};

struct Mutex;

impl Mutex {
    /// Dependent item with (potential) drop glue to disable NLL.
    fn lock(&self) -> impl '_ + Display {
        42
    }
}

struct Stderr();

impl Stderr {
    /// A "lending" `write_fmt` method. See:
    /// https://docs.rs/async-std/1.12.0/async_std/io/prelude/trait.WriteExt.html#method.write_fmt
    fn write_fmt(&mut self, _args: fmt::Arguments) -> &() { &() }
}

fn early_drop_for_format_args_temporaries() {
    let mut out = Stderr();

    let _write = {
        let mutex = Mutex;
        write!(out, "{}", mutex.lock()) /* no semicolon */
    };

    let _writeln = {
        let mutex = Mutex;
        writeln!(out, "{}", mutex.lock()) /* no semicolon */
    };
}

fn late_drop_for_receiver() {
    let mutex = Mutex;
    drop(write!(&mut Stderr(), "{}", mutex.lock()));
    drop(writeln!(&mut Stderr(), "{}", mutex.lock()));
}

fn two_phased_borrows_retrocompat(w: (&mut Stderr, i32)) {
    write!(w.0, "{}", w.1);
    writeln!(w.0, "{}", w.1);
    struct Struct<W> {
        w: W,
        len: i32
    }
    let s = (Struct { w: (w.0, ), len: w.1 }, );
    write!(s.0.w.0, "{}", s.0.len);
    writeln!(s.0.w.0, "{}", s.0.len);
}
