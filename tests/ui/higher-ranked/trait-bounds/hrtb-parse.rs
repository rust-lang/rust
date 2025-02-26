//@ run-pass
// Test that we can parse all the various places that a `for` keyword
// can appear representing universal quantification.


#![allow(unused_variables)]
#![allow(dead_code)]

trait Get<A,R> {
    fn get(&self, arg: A) -> R;
}

// Parse HRTB with explicit `for` in a where-clause:

fn foo00<T>(t: T)
    where T : for<'a> Get<&'a i32, &'a i32>
{
}

fn foo01<T: for<'a> Get<&'a i32, &'a i32>>(t: T)
{
}

// Parse HRTB with explicit `for` in various sorts of types:

fn foo10(t: Box<dyn for<'a> Get<i32, i32>>) { }
fn foo11(t: Box<dyn for<'a> Fn(i32) -> i32>) { }

fn foo20(t: for<'a> fn(i32) -> i32) { }
fn foo21(t: for<'a> unsafe fn(i32) -> i32) { }
fn foo22(t: for<'a> extern "C" fn(i32) -> i32) { }
fn foo23(t: for<'a> unsafe extern "C" fn(i32) -> i32) { }

fn main() {
}
