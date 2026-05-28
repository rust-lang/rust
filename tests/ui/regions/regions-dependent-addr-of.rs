//@ run-pass
// Test lifetimes are linked properly when we create dependent region pointers.
// Issue #3148.

#![feature(box_patterns)]

struct A {
    value: B
}

struct B {
    v1: isize,
    v2: [isize; 3],
    v3: Vec<isize> ,
    v4: C,
    v5: Box<C>,
    v6: Option<C>
}

#[derive(Copy, Clone)]
struct C {
    f: isize
}

fn get_v1(a: &A) -> &isize {
    // Region inferencer must deduce that &v < L2 < L1
    let foo = &a.value; // L1
    &foo.v1             // L2
}

fn get_v2(a: &A, i: usize) -> &isize {
    let foo = &a.value;
    &foo.v2[i]
}

fn get_v3(a: &A, i: usize) -> &isize {
    let foo = &a.value;
    &foo.v3[i]
}

fn get_v4(a: &A, _i: usize) -> &isize {
    let foo = &a.value;
    &foo.v4.f
}

fn get_v5(a: &A, _i: usize) -> &isize {
    let foo = &a.value;
    &foo.v5.f
}

fn get_v6_a(a: &A, _i: usize) -> &isize {
    match a.value.v6 {
        Some(ref v) => &v.f,
        None => panic!()
    }
}

fn get_v6_b(a: &A, _i: usize) -> &isize {
    match *a {
        A { value: B { v6: Some(ref v), .. } } => &v.f,
        _ => panic!()
    }
}

fn get_v6_c(a: &A, _i: usize) -> &isize {
    match a {
        &A { value: B { v6: Some(ref v), .. } } => &v.f,
        _ => panic!()
    }
}

fn get_v5_ref(a: &A, _i: usize) -> &isize {
    match &a.value {
        &B {v5: box C {f: ref v}, ..} => v
    }
}

pub fn main() {
    let a = A {value: B {v1: 22,
                         v2: [23, 24, 25],
                         v3: vec![26, 27, 28],
                         v4: C { f: 29 },
                         v5: Box::new(C { f: 30 }),
                         v6: Some(C { f: 31 })}};

    let p = get_v1(&a);
    assert_eq!(*p, a.value.v1);

    let p = get_v2(&a, 1);
    assert_eq!(*p, a.value.v2[1]);

    let p = get_v3(&a, 1);
    assert_eq!(*p, a.value.v3[1]);

    let p = get_v4(&a, 1);
    assert_eq!(*p, a.value.v4.f);

    let p = get_v5(&a, 1);
    assert_eq!(*p, a.value.v5.f);

    let p = get_v6_a(&a, 1);
    assert_eq!(*p, a.value.v6.unwrap().f);

    let p = get_v6_b(&a, 1);
    assert_eq!(*p, a.value.v6.unwrap().f);

    let p = get_v6_c(&a, 1);
    assert_eq!(*p, a.value.v6.unwrap().f);

    let p = get_v5_ref(&a, 1);
    assert_eq!(*p, a.value.v5.f);
}
