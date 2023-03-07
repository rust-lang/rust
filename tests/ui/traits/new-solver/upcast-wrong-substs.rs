// compile-flags: -Ztrait-solver=next

#![feature(trait_upcasting)]

trait Foo: Bar<i32> + Bar<u32> {}

trait Bar<T> {}

fn main() {
    let x: &dyn Foo = todo!();
    let y: &dyn Bar<usize> = x;
    //~^ ERROR mismatched types
}
