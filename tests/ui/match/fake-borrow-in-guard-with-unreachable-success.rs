//! Regression test for <https://github.com/rust-lang/rust/issues/161578>: the fake borrow on `x`
//! below was ignored previously because the fake read keeping it live was unreachable.

fn main() {
    let mut x: Option<Box<u64>> = Some(Box::new(7));
    match x {
        Some(_) if { x = None; false } && return => {}
        //~^ ERROR: cannot assign `x` in match guard
        Some(b) => println!("{b}"),
        None => println!("none"),
    }
}
