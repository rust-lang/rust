//@ compile-flags: -Zextra-const-ub-checks

struct A {
    f: _, //~ERROR: not allowed
}

// FIXME: the error message makes no sense
static B: &A = B; //~ERROR: access itself

fn main() {}
