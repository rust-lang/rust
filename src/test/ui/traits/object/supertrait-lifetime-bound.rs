trait Foo: 'static { }

trait Bar<T>: Foo { }

fn test1<T: ?Sized + Bar<S>, S>() { }

fn test2<'a>() {
    // Here: the type `dyn Bar<&'a u32>` references `'a`,
    // and so it does not outlive `'static`.
    test1::<dyn Bar<&'a u32>, _>();
    //~^ ERROR the type `(dyn Bar<&'a u32> + 'static)` does not fulfill the required lifetime
}

fn main() { }
