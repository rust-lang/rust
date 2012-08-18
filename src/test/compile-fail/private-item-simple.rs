// xfail-test
// xfail-fast

// This is xfail'd because two errors are reported instead of one.

mod a {
    priv fn f() {}
}

fn main() {
    a::f(); //~ ERROR unresolved name
}

