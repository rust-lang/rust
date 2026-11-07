// Regression test for https://github.com/rust-lang/rust/issues/154350

fn func<'a>(f: impl FnOnce(fn(&'a i32)), x: fn(&'static i32)) {
    f(x);
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
