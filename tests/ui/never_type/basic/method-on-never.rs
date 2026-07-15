//@ check-pass
// Regression test for https://github.com/rust-lang/rust/issues/143349

#![feature(never_type)]

trait Trait {
    fn method(&self);
}
impl Trait for ! {
    fn method(&self) {
        todo!()
    }
}

struct Adhoc;
struct Error;

#[doc(hidden)]
trait AdhocKind: Sized {
    #[inline]
    fn anyhow_kind(&self) -> Adhoc {
        Adhoc
    }
}

impl<T> AdhocKind for &T where T: ?Sized + Send + Sync + 'static {}

impl Adhoc {
    #[cold]
    fn new<M>(self, message: M) -> Error
    where
        M: Send + Sync + 'static,
    {
        Error
    }
}

fn temp<T>() -> Result<T, ()> { todo!() }

fn main() -> Result<(), ()> {
    let x = loop {};
    x.method();
    //~^ WARN [method_call_on_diverging_infer_var]
    //~| WARN previously accepted

    { loop {} }.method();
    //~^ WARN [method_call_on_diverging_infer_var]
    //~| WARN previously accepted

    let e = match loop {} {
        y => y.method(),
        //~^ WARN [method_call_on_diverging_infer_var]
        //~| WARN previously accepted
    };

    let error = match loop {} {
        error => (&error).anyhow_kind().new(error),
        //~^ WARN [method_call_on_diverging_infer_var]
        //~| WARN previously accepted
    };

    let res = temp()?;
    res.method();
    //~^ WARN [method_call_on_diverging_infer_var]
    //~| WARN previously accepted
}
