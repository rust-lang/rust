//@ revisions: may_dangle may_not_dangle
//@[may_dangle] check-pass
//@ edition: 2018

// Ensure that if a coroutine's interior has no drop types then we don't require the upvars to
// be *use-live*, but instead require them to be *drop-live*. In this case, `Droppy<&'?0 ()>`
// does not require that `'?0` is live for drops since the parameter is `#[may_dangle]` in
// the may_dangle revision, but not in the may_not_dangle revision.

#![feature(dropck_eyepatch)]

struct Droppy<T>(T);

#[cfg(may_dangle)]
unsafe impl<#[may_dangle] T> Drop for Droppy<T> {
    fn drop(&mut self) {
        // This does not use `T` of course.
    }
}

#[cfg(may_not_dangle)]
impl<T> Drop for Droppy<T> {
    fn drop(&mut self) {}
}

fn main() {
    let drop_me;
    let fut;
    {
        let y = ();
        drop_me = Droppy(&y);
        //[may_not_dangle]~^ ERROR `y` does not live long enough
        fut = async {
            std::mem::drop(drop_me);
        };
    }
}
