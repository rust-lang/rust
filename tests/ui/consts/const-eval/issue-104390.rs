fn f1() -> impl Sized { & 2E } //~ ERROR expected at least one digit in exponent
fn f2() -> impl Sized { && 2E } //~ ERROR expected at least one digit in exponent
fn f3() -> impl Sized { &'a 2E } //~ ERROR expected at least one digit in exponent
//~^ ERROR borrow expressions cannot be annotated with lifetimes
fn f4() -> impl Sized { &'static 2E } //~ ERROR expected at least one digit in exponent
//~^ ERROR borrow expressions cannot be annotated with lifetimes
fn f5() -> impl Sized { *& 2E } //~ ERROR expected at least one digit in exponent
fn f6() -> impl Sized { &'_ 2E } //~ ERROR expected at least one digit in exponent
//~^ ERROR borrow expressions cannot be annotated with lifetimes
fn main() {}
