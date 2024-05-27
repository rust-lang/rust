//! Show an uninformative diagnostic that we could possibly improve in the future

trait Trait<T> {}

impl<T, U> Trait<T> for U {}

fn hello() -> &'static (dyn Trait<impl Sized> + Send) {
    //~^ ERROR: type annotations needed
    if false {
        let x = hello();
        let _: &'static dyn Trait<()> = &x;
        //^ Note the extra `&`, paired with the blanket impl causing
        // `impl Sized` to never get a hidden type registered.
    }
    todo!()
}

fn main() {}
