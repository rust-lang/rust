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

struct Wrapper1<T: Foo>(#[allow(unused_tuple_struct_fields)] <T as Foo>::Bar);
struct Wrapper2<T: Foo>(#[allow(unused_tuple_struct_fields)] <Wrapper1<T> as Pointee>::Metadata);

fn main() {
    let _: Wrapper2<()> = Wrapper2(());
    let _ = Layout::new::<Wrapper2<()>>();
}
