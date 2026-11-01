//! Test that we allow unsizing `Foo<[Opaque; N]>` to `Foo<[Concrete]>`.

//@check-pass

struct Foo<T: ?Sized>(T);

fn hello() -> Foo<[impl Sized; 2]> {
    if false {
        let x = hello();
        let _: &Foo<[i32]> = &x;
    }
    todo!()
}

fn main() {}
