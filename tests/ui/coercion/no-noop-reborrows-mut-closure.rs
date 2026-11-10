// Ideally, this would pass.
//@ revisions: fail0 fail1 fail2

struct NotCopy<'a>(&'a mut bool);

fn test<'a>(m: &'a mut bool) {
    (move || {
        let r = (NotCopy(m),);
        drop(r.0);

        // `r.0` is now moved  out of,
        // but `r` is not fully moved

        let _ = r.0.0;
        #[cfg(fail0)]
        let _: &mut _ = r.0.0;
        //[fail0]~^ ERROR borrow of moved value: `r.0`
        #[cfg(fail1)]
        let _: &mut bool = r.0.0;
        //[fail1]~^ ERROR borrow of moved value: `r.0`
        #[cfg(fail2)]
        let _: &'a mut bool = r.0.0;
        //[fail2]~^ ERROR borrow of moved value: `r.0`
        //[fail2]~| ERROR lifetime may not live long enough
    })();
}

fn main() {
    test(&mut false);
}
