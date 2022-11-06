// known-bug: unknown
// FIXME(chalk): Chalk needs support for the Tuple trait
// compile-flags: -Z chalk

fn main() -> () {
    let t = || {};
    t();

    let mut a = 0;
    let mut b = move || {
        a = 1;
    };
    b();

    let mut c = b;

    c();
    b();

    let mut a = 0;
    let mut b = || {
        a = 1;
    };
    b();

    let mut c = b;

    c();
    b(); // FIXME: reenable when this is fixed ~ ERROR

    // FIXME(chalk): this doesn't quite work
    /*
    let b = |c| {
        c
    };

    let a = &32;
    b(a);
    */
}
