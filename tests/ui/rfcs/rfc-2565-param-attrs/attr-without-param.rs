//@ edition:2015
#[cfg(false)]
impl S {
    fn f(#[attr]) {} //~ ERROR expected parameter name, found `)`
}

#[cfg(false)]
impl T for S {
    fn f(#[attr]) {} //~ ERROR expected parameter name, found `)`
}

#[cfg(false)]
trait T {
    fn f(#[attr]); //~ ERROR expected type, found `)`
}

fn main() {}
