#![feature(return_type_notation)]

trait Trait {
    fn method() -> impl Sized;
}

struct DoesntWork;
impl Trait for DoesntWork {
    fn method() -> impl Sized {
        std::ptr::null_mut::<()>()
        // This isn't `Send`.
    }
}

fn test<T: Trait>()
where
    T::method(..): Send,
{
}

fn main() {
    test::<DoesntWork>();
    //~^ ERROR `*mut ()` cannot be sent between threads safely
}
