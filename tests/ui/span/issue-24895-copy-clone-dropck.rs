// Check that one cannot subvert Drop Check rule via a user-defined
// Clone implementation.

#![allow(unused_variables, unused_assignments)]

struct D<T:Copy>(T, &'static str);

#[derive(Copy)]
struct S<'a>(&'a D<i32>, &'static str);
impl<'a> Clone for S<'a> {
    fn clone(&self) -> S<'a> {
        println!("cloning `S(_, {})` and thus accessing: {}", self.1, (self.0).0);
        S(self.0, self.1)
    }
}

impl<T:Copy> Drop for D<T> {
    fn drop(&mut self) {
        println!("calling Drop for {}", self.1);
        let _call = self.0.clone();
    }
}

fn main() {
    let (d2, d1);
    d1 = D(34, "d1");
    d2 = D(S(&d1, "inner"), "d2");
}
//~^^ ERROR `d1` does not live long enough
