// Test that assignments to an `&mut` pointer which is found in a
// borrowed (but otherwise non-aliasable) location is illegal.

struct S<'a> {
    pointer: &'a mut isize
}

fn a(s: &S) {
    *s.pointer += 1; //~ ERROR cannot assign
}

fn b(s: &mut S) {
    *s.pointer += 1;
}

fn c(s: & &mut S) {
    *s.pointer += 1; //~ ERROR cannot assign
}

fn main() {}
