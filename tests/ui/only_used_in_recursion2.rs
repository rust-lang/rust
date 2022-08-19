#![warn(clippy::only_used_in_recursion)]

fn _with_inner(flag: u32, a: u32, b: u32) -> usize {
    fn inner(flag: u32, a: u32) -> u32 {
        if flag == 0 { 0 } else { inner(flag, a) }
    }

    let x = inner(flag, a);
    if flag == 0 { 0 } else { _with_inner(flag, a, b + x) }
}

fn _with_closure(a: Option<u32>, b: u32, f: impl Fn(u32, u32) -> Option<u32>) -> u32 {
    if let Some(x) = a.and_then(|x| f(x, x)) {
        _with_closure(Some(x), b, f)
    } else {
        0
    }
}

// Issue #8560
trait D {
    fn foo(&mut self, arg: u32) -> u32;
}

mod m {
    pub struct S(u32);
    impl S {
        pub fn foo(&mut self, arg: u32) -> u32 {
            arg + self.0
        }
    }
}

impl D for m::S {
    fn foo(&mut self, arg: u32) -> u32 {
        self.foo(arg)
    }
}

// Issue #8782
fn only_let(x: u32) {
    let y = 10u32;
    let _z = x * y;
}

trait E<T: E<()>> {
    fn method(flag: u32, a: usize) -> usize {
        if flag == 0 {
            0
        } else {
            <T as E<()>>::method(flag - 1, a)
        }
    }
}

impl E<()> for () {
    fn method(flag: u32, a: usize) -> usize {
        if flag == 0 { 0 } else { a }
    }
}

fn overwritten_param(flag: u32, mut a: usize) -> usize {
    if flag == 0 {
        return 0;
    } else if flag > 5 {
        a += flag as usize;
    } else {
        a = 5;
    }
    overwritten_param(flag, a)
}

fn field_direct(flag: u32, mut a: (usize,)) -> usize {
    if flag == 0 {
        0
    } else {
        a.0 += 5;
        field_direct(flag - 1, a)
    }
}

fn field_deref(flag: u32, a: &mut Box<(usize,)>) -> usize {
    if flag == 0 {
        0
    } else {
        a.0 += 5;
        field_deref(flag - 1, a)
    }
}

fn main() {}
