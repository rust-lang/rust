// Fixed by #67071
// aux-build: issue_66868_closure_typeck.rs
// edition:2018

extern crate issue_66868_closure_typeck;

pub fn g<T>(task: T)
where
    T: Send,
{
}

fn main() {
    g(issue_66868_closure_typeck::f()); //~ ERROR: cannot be sent between threads safely
}
