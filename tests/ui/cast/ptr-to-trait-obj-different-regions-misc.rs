//@ check-fail

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


fn main() {}
