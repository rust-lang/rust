//! Test that we allow unsizing `Trait<Concrete>` to `Trait<Opaque>` and vice versa

//@ check-pass

trait Trait<T> {}

impl<T, U> Trait<T> for U {}

fn hello() -> &'static (dyn Trait<impl Sized> + Send) {
    if false {
        let x = hello();
        let _: &'static dyn Trait<()> = x;
    }
    todo!()
}

fn bye() -> &'static dyn Trait<impl Sized> {
    if false {
        let mut x = bye();
        let y: &'static (dyn Trait<()> + Send) = &();
        x = y;
    }
    todo!()
}

fn main() {}
