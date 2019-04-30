// ignore-windows
// ignore-emscripten no threads support
// exec-env:RUSTC_LOG=debug

use std::cell::Cell;
use std::fmt;
use std::thread;

struct Foo(Cell<isize>);

impl fmt::Debug for Foo {
    fn fmt(&self, _fmt: &mut fmt::Formatter) -> fmt::Result {
        let Foo(ref f) = *self;
        assert_eq!(f.get(), 0);
        f.set(1);
        Ok(())
    }
}

pub fn main() {
    thread::spawn(move|| {
        let mut f = Foo(Cell::new(0));
        println!("{:?}", f);
        let Foo(ref mut f) = f;
        assert_eq!(f.get(), 1);
    }).join().ok().unwrap();
}
