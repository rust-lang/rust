//! Coroutines mark locals as dead on unwind paths, so the borrow checker rejects
//! a local being dead while a destructor that uses a borrow of it is still pending
//! on such a path. Temporaries that are only ever accessed through a dereference
//! are not marked as dead on unwind paths in coroutines, which must not change
//! what the borrow checker accepts.
//!
//! See <https://github.com/rust-lang/rust/issues/115327> and
//! <https://github.com/rust-lang/rust/issues/147875>.
//@ edition: 2024

#![allow(unused)]

use std::fmt::Display;

struct Wrap<T: Display>(T);

impl<T: Display> Drop for Wrap<T> {
    fn drop(&mut self) {
        println!("{}", self.0);
    }
}

fn id<T>(t: T) -> T {
    t
}

async fn direct_borrow() {
    let x;
    {
        let y = 1;
        x = Wrap(&y);
        //~^ ERROR `y` does not live long enough
        panic!();
    }
}

async fn borrow_through_dereferenced_temp() {
    let x;
    {
        let y = 1;
        x = Wrap(&*id(&y));
        //~^ ERROR `y` does not live long enough
        panic!();
    }
}

async fn borrow_through_dereferenced_temp_across_await() {
    let x;
    {
        let y = 1;
        x = Wrap(&*id(&y));
        //~^ ERROR `y` does not live long enough
        async {}.await;
        panic!();
    }
}

async fn borrow_of_temp_through_dereferenced_temp() {
    let x;
    {
        x = Wrap(&*id(&String::new()));
        //~^ ERROR temporary value dropped while borrowed
        panic!();
    }
}

async fn borrows_through_dereferenced_temps_in_vec() {
    let x;
    {
        let y = String::new();
        x = vec![Wrap(&*id(y.as_str())), Wrap(&*id(&*y.clone()))];
        //~^ ERROR `y` does not live long enough
        //~| ERROR temporary value dropped while borrowed
        panic!();
    }
}

async fn mutable_borrow_through_dereferenced_temp() {
    let x;
    {
        let mut y = 1;
        x = Wrap(&mut *id(&mut y));
        //~^ ERROR `y` does not live long enough
        async {}.await;
        panic!();
    }
}

fn panicking_guard(_: &i32) -> bool {
    panic!()
}

// A match guard fakely borrows the dereferenced temporary itself.
async fn guard_on_dereferenced_temp() {
    let x;
    {
        let y = 1;
        match *id(&y) {
            //~^ ERROR `y` does not live long enough
            ref z if panicking_guard(z) => x = Wrap(z),
            _ => panic!(),
        }
        panic!();
    }
}

// Functions that are not coroutines do not mark locals as dead on unwind paths.
fn not_a_coroutine() {
    let x;
    {
        let y = 1;
        x = Wrap(&*id(&y));
        panic!();
    }
}

fn main() {}
