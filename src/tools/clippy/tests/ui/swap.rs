//@aux-build: macro_rules.rs

#![allow(
    clippy::disallowed_names,
    clippy::no_effect,
    clippy::redundant_clone,
    clippy::let_and_return,
    clippy::useless_vec,
    clippy::redundant_locals
)]

struct Foo(u32);

#[derive(Clone)]
struct Bar {
    a: u32,
    b: u32,
}

fn field() {
    let mut bar = Bar { a: 1, b: 2 };

    let temp = bar.a;
    //~^ manual_swap
    bar.a = bar.b;
    bar.b = temp;

    let mut baz = vec![bar.clone(), bar.clone()];
    let temp = baz[0].a;
    baz[0].a = baz[1].a;
    baz[1].a = temp;
}

fn array() {
    let mut foo = [1, 2];
    let temp = foo[0];
    //~^ manual_swap
    foo[0] = foo[1];
    foo[1] = temp;

    foo.swap(0, 1);
}

fn slice() {
    let foo = &mut [1, 2];
    let temp = foo[0];
    //~^ manual_swap
    foo[0] = foo[1];
    foo[1] = temp;

    foo.swap(0, 1);
}

fn unswappable_slice() {
    let foo = &mut [vec![1, 2], vec![3, 4]];
    let temp = foo[0][1];
    foo[0][1] = foo[1][0];
    foo[1][0] = temp;

    // swap(foo[0][1], foo[1][0]) would fail
    // this could use split_at_mut and mem::swap, but that is not much simpler.
}

fn vec() {
    let mut foo = vec![1, 2];
    let temp = foo[0];
    //~^ manual_swap
    foo[0] = foo[1];
    foo[1] = temp;

    foo.swap(0, 1);
}

fn xor_swap_locals() {
    // This is an xor-based swap of local variables.
    let mut a = 0;
    let mut b = 1;
    a ^= b;
    //~^ manual_swap
    b ^= a;
    a ^= b;
}

fn xor_field_swap() {
    // This is an xor-based swap of fields in a struct.
    let mut bar = Bar { a: 0, b: 1 };
    bar.a ^= bar.b;
    //~^ manual_swap
    bar.b ^= bar.a;
    bar.a ^= bar.b;
}

fn xor_slice_swap() {
    // This is an xor-based swap of a slice
    let foo = &mut [1, 2];
    foo[0] ^= foo[1];
    //~^ manual_swap
    foo[1] ^= foo[0];
    foo[0] ^= foo[1];
}

fn xor_no_swap() {
    // This is a sequence of xor-assignment statements that doesn't result in a swap.
    let mut a = 0;
    let mut b = 1;
    let mut c = 2;
    a ^= b;
    b ^= c;
    a ^= c;
    c ^= a;
}

fn xor_unswappable_slice() {
    let foo = &mut [vec![1, 2], vec![3, 4]];
    foo[0][1] ^= foo[1][0];
    foo[1][0] ^= foo[0][0];
    foo[0][1] ^= foo[1][0];

    // swap(foo[0][1], foo[1][0]) would fail
    // this could use split_at_mut and mem::swap, but that is not much simpler.
}

fn distinct_slice() {
    let foo = &mut [vec![1, 2], vec![3, 4]];
    let bar = &mut [vec![1, 2], vec![3, 4]];
    let temp = foo[0][1];
    //~^ manual_swap
    foo[0][1] = bar[1][0];
    bar[1][0] = temp;
}

#[rustfmt::skip]
fn main() {

    let mut a = 42;
    let mut b = 1337;

    a = b;
    //~^ almost_swapped
    b = a;

    ; let t = a;
    //~^ manual_swap
    a = b;
    b = t;

    let mut c = Foo(42);

    c.0 = a;
    //~^ almost_swapped
    a = c.0;

    ; let t = c.0;
    //~^ manual_swap
    c.0 = a;
    a = t;

    let a = b;
    //~^ almost_swapped
    let b = a;

    let mut c = 1;
    let mut d = 2;
    d = c;
    //~^ almost_swapped
    c = d;

    let mut b = 1;
    let a = b;
    //~^ almost_swapped
    b = a;

    let b = 1;
    let a = 2;

    let t = b;
    let b = a;
    let a = t;

    let mut b = 1;
    let mut a = 2;

    let t = b;
    //~^ manual_swap
    b = a;
    a = t;
}

fn issue_8154() {
    struct S1 {
        x: i32,
        y: i32,
    }
    struct S2(S1);
    struct S3<'a, 'b>(&'a mut &'b mut S1);

    impl std::ops::Deref for S2 {
        type Target = S1;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl std::ops::DerefMut for S2 {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    // Don't lint. `s.0` is mutably borrowed by `s.x` and `s.y` via the deref impl.
    let mut s = S2(S1 { x: 0, y: 0 });
    let t = s.x;
    s.x = s.y;
    s.y = t;

    // Accessing through a mutable reference is fine
    let mut s = S1 { x: 0, y: 0 };
    let mut s = &mut s;
    let s = S3(&mut s);
    let t = s.0.x;
    //~^ manual_swap
    s.0.x = s.0.y;
    s.0.y = t;
}

const fn issue_9864(mut u: u32) -> u32 {
    let mut v = 10;

    let temp = u;
    u = v;
    v = temp;
    u + v
}

#[macro_use]
extern crate macro_rules;

const fn issue_10421(x: u32) -> u32 {
    issue_10421!();
    let a = x;
    let a = a;
    let a = a;
    a
}
