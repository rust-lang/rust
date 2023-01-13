// Checks that certain traits for which we don't want to suggest borrowing
// are blacklisted and don't cause the suggestion to be issued.

#![feature(generators)]

fn f_copy<T: Copy>(t: T) {}
fn f_clone<T: Clone>(t: T) {}
fn f_unpin<T: Unpin>(t: T) {}
fn f_sized<T: Sized>(t: T) {}
fn f_send<T: Send>(t: T) {}

struct S;

fn main() {
    f_copy("".to_string()); //~ ERROR: the trait bound `String: Copy` is not satisfied [E0277]
    f_clone(S); //~ ERROR: the trait bound `S: Clone` is not satisfied [E0277]
    f_unpin(static || { yield; });
    //~^ ERROR: cannot be unpinned [E0277]

    let cl = || ();
    let ref_cl: &dyn Fn() -> () = &cl;
    f_sized(*ref_cl);
    //~^ ERROR: the size for values of type `dyn Fn()` cannot be known at compilation time [E0277]

    use std::rc::Rc;
    let rc = Rc::new(0);
    f_send(rc); //~ ERROR: `Rc<{integer}>` cannot be sent between threads safely [E0277]
}
