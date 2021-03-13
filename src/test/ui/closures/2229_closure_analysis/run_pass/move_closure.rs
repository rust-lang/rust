// run-pass

// Test that move closures compile properly with `capture_disjoint_fields` enabled.

#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete

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

struct A<'a>(&'a mut String,  &'a mut String);
// Test that reborrowing works as expected for move closures
// by attempting a disjoint capture through a reference.
fn disjoint_via_ref() {
    let mut x = String::new();
    let mut y = String::new();

    let mut a = A(&mut x, &mut y);
    let a = &mut a;

    let mut c1 = move || {
        a.0.truncate(0);
    };

    let mut c2 = move || {
        a.1.truncate(0);
    };

    c1();
    c2();
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

fn main() {
    simple_ref();
    struct_contains_ref_to_another_struct();
    no_ref();
    no_ref_nested();

    disjoint_via_ref();
    data_moved_but_not_fn_once();
}
