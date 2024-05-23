//! Test that we do not allow unsizing `Foo<[Opaque; N]>` to `Foo<[Concrete]>`.

struct Foo<T: ?Sized>(T);

fn hello() -> Foo<[impl Sized; 2]> {
    if false {
        let x = hello();
        let _: &Foo<[i32]> = &x;
        //~^ ERROR: mismatched types
    }
    todo!()
}

fn main() {}
