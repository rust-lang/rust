//@ check-fail
//@ dont-require-annotations: NOTE

trait Trait<'a> {}

fn change_lt<'a, 'b>(x: *mut dyn Trait<'a>) -> *mut dyn Trait<'b> {
    x as _ //~ error: lifetime may not live long enough
           //~| error: lifetime may not live long enough
}

fn change_lt_ab<'a: 'b, 'b>(x: *mut dyn Trait<'a>) -> *mut dyn Trait<'b> {
    x as _ //~ error: lifetime may not live long enough
}

fn change_lt_ba<'a, 'b: 'a>(x: *mut dyn Trait<'a>) -> *mut dyn Trait<'b> {
    x as _ //~ error: lifetime may not live long enough
}

fn change_lt_hr<'a>(x: *mut dyn Trait<'a>) -> *mut dyn for<'b> Trait<'b> {
    x as _ //~ error: lifetime may not live long enough
    //~^ error: mismatched types
    //~| NOTE one type is more general than the other
}

trait Assocked {
    type Assoc: ?Sized;
}

fn change_assoc_0<'a, 'b>(
    x: *mut dyn Assocked<Assoc = dyn Send + 'a>,
) -> *mut dyn Assocked<Assoc = dyn Send + 'b> {
    x as _ //~ error: lifetime may not live long enough
           //~| error: lifetime may not live long enough
}

fn change_assoc_1<'a, 'b>(
    x: *mut dyn Assocked<Assoc = dyn Trait<'a>>,
) -> *mut dyn Assocked<Assoc = dyn Trait<'b>> {
    x as _ //~ error: lifetime may not live long enough
           //~| error: lifetime may not live long enough
}

// This tests the default borrow check error, without the special casing for return values.
fn require_static(_: *const dyn Trait<'static>) {}
fn extend_to_static<'a>(ptr: *const dyn Trait<'a>) {
    require_static(ptr as _) //~ error: borrowed data escapes outside of function
}

fn main() {}
