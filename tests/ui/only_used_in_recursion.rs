#![warn(clippy::only_used_in_recursion)]

fn simple(a: usize, b: usize) -> usize {
    if a == 0 { 1 } else { simple(a - 1, b) }
}

fn with_calc(a: usize, b: isize) -> usize {
    if a == 0 { 1 } else { with_calc(a - 1, -b + 1) }
}

fn tuple((a, b): (usize, usize)) -> usize {
    if a == 0 { 1 } else { tuple((a - 1, b + 1)) }
}

fn let_tuple(a: usize, b: usize) -> usize {
    let (c, d) = (a, b);
    if c == 0 { 1 } else { let_tuple(c - 1, d + 1) }
}

fn array([a, b]: [usize; 2]) -> usize {
    if a == 0 { 1 } else { array([a - 1, b + 1]) }
}

fn index(a: usize, mut b: &[usize], c: usize) -> usize {
    if a == 0 { 1 } else { index(a - 1, b, c + b[0]) }
}

fn break_(a: usize, mut b: usize, mut c: usize) -> usize {
    let c = loop {
        b += 1;
        c += 1;
        if c == 10 {
            break b;
        }
    };

    if a == 0 { 1 } else { break_(a - 1, c, c) }
}

// this has a side effect
fn mut_ref(a: usize, b: &mut usize) -> usize {
    *b = 1;
    if a == 0 { 1 } else { mut_ref(a - 1, b) }
}

fn mut_ref2(a: usize, b: &mut usize) -> usize {
    let mut c = *b;
    if a == 0 { 1 } else { mut_ref2(a - 1, &mut c) }
}

fn not_primitive(a: usize, b: String) -> usize {
    if a == 0 { 1 } else { not_primitive(a - 1, b) }
}

// this doesn't have a side effect,
// but `String` is not primitive.
fn not_primitive_op(a: usize, b: String, c: &str) -> usize {
    if a == 1 { 1 } else { not_primitive_op(a, b + c, c) }
}

struct A;

impl A {
    fn method(a: usize, b: usize) -> usize {
        if a == 0 { 1 } else { A::method(a - 1, b - 1) }
    }

    fn method2(&self, a: usize, b: usize) -> usize {
        if a == 0 { 1 } else { self.method2(a - 1, b + 1) }
    }
}

trait B {
    fn hello(a: usize, b: usize) -> usize;

    fn hello2(&self, a: usize, b: usize) -> usize;
}

impl B for A {
    fn hello(a: usize, b: usize) -> usize {
        if a == 0 { 1 } else { A::hello(a - 1, b + 1) }
    }

    fn hello2(&self, a: usize, b: usize) -> usize {
        if a == 0 { 1 } else { self.hello2(a - 1, b + 1) }
    }
}

trait C {
    fn hello(a: usize, b: usize) -> usize {
        if a == 0 { 1 } else { Self::hello(a - 1, b + 1) }
    }

    fn hello2(&self, a: usize, b: usize) -> usize {
        if a == 0 { 1 } else { self.hello2(a - 1, b + 1) }
    }
}

fn ignore(a: usize, _: usize) -> usize {
    if a == 1 { 1 } else { ignore(a - 1, 0) }
}

fn ignore2(a: usize, _b: usize) -> usize {
    if a == 1 { 1 } else { ignore2(a - 1, _b) }
}

fn f1(a: u32) -> u32 {
    a
}

fn f2(a: u32) -> u32 {
    f1(a)
}

fn inner_fn(a: u32) -> u32 {
    fn inner_fn(a: u32) -> u32 {
        a
    }
    inner_fn(a)
}

fn main() {}
