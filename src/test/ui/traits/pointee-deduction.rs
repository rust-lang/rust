// run-pass

#![feature(ptr_metadata)]

use std::alloc::Layout;
use std::ptr::Pointee;

trait Foo {
    type Bar;
}

impl Foo for () {
    type Bar = ();
}

struct Wrapper1<T: Foo>(<T as Foo>::Bar);
struct Wrapper2<T: Foo>(<Wrapper1<T> as Pointee>::Metadata);

fn main() {
    let _: Wrapper2<()> = Wrapper2(());
    let _ = Layout::new::<Wrapper2<()>>();
}
