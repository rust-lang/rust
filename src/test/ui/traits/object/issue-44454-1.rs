// Taken from https://github.com/rust-lang/rust/issues/44454#issue-256435333

trait Animal<X>: 'static {}

fn foo<Y, X>()
where
    Y: Animal<X> + ?Sized,
{
    // `Y` implements `Animal<X>` so `Y` is 'static.
    baz::<Y>()
}

fn bar<'a>(_arg: &'a i32) {
    foo::<dyn Animal<&'a i32>, &'a i32>() //~ ERROR: lifetime may not live long enough
}

fn baz<T: 'static + ?Sized>() {}

fn main() {
    let a = 5;
    bar(&a);
}
