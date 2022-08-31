#![warn(clippy::only_used_in_recursion)]

fn _simple(x: u32) -> u32 {
    x
}

fn _simple2(x: u32) -> u32 {
    _simple(x)
}

fn _one_unused(flag: u32, a: usize) -> usize {
    if flag == 0 { 0 } else { _one_unused(flag - 1, a) }
}

fn _two_unused(flag: u32, a: u32, b: i32) -> usize {
    if flag == 0 { 0 } else { _two_unused(flag - 1, a, b) }
}

fn _with_calc(flag: u32, a: i64) -> usize {
    if flag == 0 {
        0
    } else {
        _with_calc(flag - 1, (-a + 10) * 5)
    }
}

// Don't lint
fn _used_with_flag(flag: u32, a: u32) -> usize {
    if flag == 0 { 0 } else { _used_with_flag(flag ^ a, a - 1) }
}

fn _used_with_unused(flag: u32, a: i32, b: i32) -> usize {
    if flag == 0 {
        0
    } else {
        _used_with_unused(flag - 1, -a, a + b)
    }
}

fn _codependent_unused(flag: u32, a: i32, b: i32) -> usize {
    if flag == 0 {
        0
    } else {
        _codependent_unused(flag - 1, a * b, a + b)
    }
}

fn _not_primitive(flag: u32, b: String) -> usize {
    if flag == 0 { 0 } else { _not_primitive(flag - 1, b) }
}

struct A;

impl A {
    fn _method(flag: usize, a: usize) -> usize {
        if flag == 0 { 0 } else { Self::_method(flag - 1, a) }
    }

    fn _method_self(&self, flag: usize, a: usize) -> usize {
        if flag == 0 { 0 } else { self._method_self(flag - 1, a) }
    }
}

trait B {
    fn method(flag: u32, a: usize) -> usize;
    fn method_self(&self, flag: u32, a: usize) -> usize;
}

impl B for A {
    fn method(flag: u32, a: usize) -> usize {
        if flag == 0 { 0 } else { Self::method(flag - 1, a) }
    }

    fn method_self(&self, flag: u32, a: usize) -> usize {
        if flag == 0 { 0 } else { self.method_self(flag - 1, a) }
    }
}

impl B for () {
    fn method(flag: u32, a: usize) -> usize {
        if flag == 0 { 0 } else { a }
    }

    fn method_self(&self, flag: u32, a: usize) -> usize {
        if flag == 0 { 0 } else { a }
    }
}

impl B for u32 {
    fn method(flag: u32, a: usize) -> usize {
        if flag == 0 { 0 } else { <() as B>::method(flag, a) }
    }

    fn method_self(&self, flag: u32, a: usize) -> usize {
        if flag == 0 { 0 } else { ().method_self(flag, a) }
    }
}

trait C {
    fn method(flag: u32, a: usize) -> usize {
        if flag == 0 { 0 } else { Self::method(flag - 1, a) }
    }

    fn method_self(&self, flag: u32, a: usize) -> usize {
        if flag == 0 { 0 } else { self.method_self(flag - 1, a) }
    }
}

fn _ignore(flag: usize, _a: usize) -> usize {
    if flag == 0 { 0 } else { _ignore(flag - 1, _a) }
}

fn main() {}
