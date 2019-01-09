// Test closure that takes two references and is supposed to return
// the first, but actually returns the second. This should fail within
// the closure.

// compile-flags:-Zborrowck=mir -Zverbose

#![feature(rustc_attrs)]

#[rustc_regions]
fn test() {
    expect_sig(|a, b| b); // ought to return `a`
    //~^ ERROR
}

fn expect_sig<F>(f: F) -> F
    where F: for<'a> FnMut(&'a i32, &i32) -> &'a i32
{
    f
}

fn deref(_p: &i32) { }

fn main() { }
