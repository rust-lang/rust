// Issue 4691: Ensure that functional-struct-update can only copy, not
// move, when the struct implements Drop.

struct B;
struct S { a: isize, b: B }
impl Drop for S { fn drop(&mut self) { } }

struct T { a: isize, mv: Box<isize> }
impl Drop for T { fn drop(&mut self) { } }

fn f(s0:S) {
    let _s2 = S{a: 2, ..s0};
    //~^ ERROR [E0509]
}

fn g(s0:T) {
    let _s2 = T{a: 2, ..s0};
    //~^ ERROR [E0509]
}

fn main() { }
