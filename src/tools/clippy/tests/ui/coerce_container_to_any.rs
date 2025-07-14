#![warn(clippy::coerce_container_to_any)]

use std::any::Any;

fn main() {
    let mut x: Box<dyn Any> = Box::new(());
    let ref_x = &x;

    f(&x);
    //~^ coerce_container_to_any

    f(ref_x);
    //~^ coerce_container_to_any

    let _: &dyn Any = &x;
    //~^ coerce_container_to_any

    let _: &dyn Any = &mut x;
    //~^ coerce_container_to_any

    let _: &mut dyn Any = &mut x;
    //~^ coerce_container_to_any

    f(&42);
    f(&Box::new(()));
    f(&Box::new(Box::new(())));
    let ref_x = &x;
    f(&**ref_x);
    f(&*x);
    let _: &dyn Any = &*x;

    // https://github.com/rust-lang/rust-clippy/issues/15045
    #[allow(clippy::needless_borrow)]
    (&x).downcast_ref::<()>().unwrap();
}

fn f(_: &dyn Any) {}
