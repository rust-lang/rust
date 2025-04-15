//@ known-bug: #139381
//@ needs-rustc-debug-assertions
trait A<'a> {
    type Assoc: ?Sized;
}

impl<'a> A<'a> for () {
    type Assoc = &'a ();
}

fn hello() -> impl for<'a> A<'a, Assoc: Into<u8> + 'static + Copy> {
    ()
}
