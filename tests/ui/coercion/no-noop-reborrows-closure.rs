// Ideally, this would pass.
//@ revisions: fail0 fail1 pass
//@[pass] run-pass

struct NotCopy(&'static bool);

fn main() {
    (|| {
        let r = (NotCopy(&true),);
        drop(r.0);

        // `r.0` is now moved  out of,
        // but `r` is not fully moved

        let _ = r.0.0;
        #[cfg(fail0)]
        let _: &_ = r.0.0;
        //[fail0]~^ ERROR borrow of moved value: `r.0`
        #[cfg(fail1)]
        let _: &bool = r.0.0;
        //[fail1]~^ ERROR borrow of moved value: `r.0`
        let _: &'static bool = r.0.0;
    })();
}
