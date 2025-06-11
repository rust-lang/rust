/*! ```compile_fail,E0373

fn bad_scope<F>(f: F)
    where F: FnOnce(&i32) + Send,
{
    rayon_core::scope(|s| {
        let x = 22;
        s.spawn(|_| f(&x)); //~ ERROR `x` does not live long enough
    });
}

fn good_scope<F>(f: F)
    where F: FnOnce(&i32) + Send,
{
    let x = 22;
    rayon_core::scope(|s| {
        s.spawn(|_| f(&x));
    });
}

fn main() {
}

``` */
