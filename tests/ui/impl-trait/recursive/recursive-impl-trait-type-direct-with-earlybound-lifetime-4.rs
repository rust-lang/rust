#![allow(unconditional_recursion)]

trait Trait<'a, 'b> {}
impl<'a, 'b, T> Trait<'a, 'b> for T {}

fn foo<'a: 'a, 'b: 'b>() -> impl Trait<'a, 'b> {
    let _: *mut &'a () = foo::<'a, 'b>();
    let _: *mut &'b () = foo::<'a, 'b>();
    //~^ ERROR
    loop {}
}

fn main() {}
