//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass
// Checks that we can infer a closure signature even if the `FnOnce` bound is
// a supertrait of the obligations we have currently registered for the Ty var.

pub trait Receive<T, E>: FnOnce(Result<T, E>) {
    fn receive(self, res: Result<T, E>);
}

impl<T, E, F: FnOnce(Result<T, E>)> Receive<T, E> for F {
    fn receive(self, res: Result<T, E>) {
        self(res)
    }
}

pub trait Async<T, E> {
    fn receive<F: Receive<T, E>>(self, f: F);
}

impl<T, E> Async<T, E> for Result<T, E> {
    fn receive<F: Receive<T, E>>(self, f: F) {
        f(self)
    }
}

pub fn main() {
    Ok::<u32, ()>(123).receive(|res| {
        res.unwrap();
    });
}
