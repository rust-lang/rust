//@ edition:2021
//@ run-pass

// Test that move closures compile properly with `capture_disjoint_fields` enabled.

#![allow(unused)]

fn simple_ref() {
    let mut s = 10;
    let ref_s = &mut s;

    let mut c = move || {
        *ref_s += 10;
    };
    c();
}

fn struct_contains_ref_to_another_struct() {
    struct S(String);
    struct T<'a>(&'a mut S);

    let mut s = S("s".into());
    let t = T(&mut s);

    let mut c = move || {
        t.0.0 = "new s".into();
    };

    c();
}

#[derive(Debug)]
struct S(String);

#[derive(Debug)]
struct T(S);

fn no_ref() {
    let mut t = T(S("s".into()));
    let mut c = move || {
        t.0.0 = "new S".into();
    };
    c();
}

fn no_ref_nested() {
    let mut t = T(S("s".into()));
    let c = || {
        println!("{:?}", t.0);
        let mut c = move || {
            t.0.0 = "new S".into();
            println!("{:?}", t.0.0);
        };
        c();
    };
    c();
}

// Test that even if a path is moved into the closure, the closure is not FnOnce
// if the path is not moved by the closure call.
fn data_moved_but_not_fn_once() {
    let x = Box::new(10i32);

    let c = move || {
        // *x has type i32 which is Copy. So even though the box `x` will be moved
        // into the closure, `x` is never moved when the closure is called, i.e. the
        // ownership stays with the closure and therefore we can call the function multiple times.
        let _x = *x;
    };

    c();
    c();
}

// Test that move closures can take ownership of Copy type
fn returned_closure_owns_copy_type_data() -> impl Fn() -> i32 {
    let x = 10;

    let c = move || x;

    c
}

fn main() {
    simple_ref();
    struct_contains_ref_to_another_struct();
    no_ref();
    no_ref_nested();

    data_moved_but_not_fn_once();

    returned_closure_owns_copy_type_data();
}
