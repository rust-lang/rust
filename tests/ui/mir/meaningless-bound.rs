//! Regression test for #140100 and #140365
//@compile-flags: -C opt-level=1 -Zvalidate-mir

fn a()
where
    b: Sized,
    //~^ ERROR cannot find type `b` in this scope
{
    println!()
}

fn f() -> &'static str
where
    Self: Sized,
    //~^ ERROR cannot find type `Self` in this scope
{
    ""
}

fn main() {}
