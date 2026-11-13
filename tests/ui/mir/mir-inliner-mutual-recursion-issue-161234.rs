//@ revisions: opt2 opt3
//@[opt2] compile-flags: -Copt-level=2
//@[opt3] compile-flags: -Copt-level=3
//@ build-pass

// Regression test for #161234
//
// The MIR inliner call-graph analysis used to cache a temporary `false`
// reachability result while walking a cycle. A different edge in the same
// cycle could later prove that the function did reach the root, leaving the
// cached result stale and allowing an invalid query cycle.

fn a() {
    b();
    c();
}

fn b() {
    a();
    d();
}

fn c() {
    d();
    b();
}

fn d() {
    c();
    a();
}

fn a_reordered() {
    c_reordered();
    b_reordered();
}

fn b_reordered() {
    d_reordered();
    a_reordered();
}

fn c_reordered() {
    b_reordered();
    d_reordered();
}

fn d_reordered() {
    a_reordered();
    c_reordered();
}

fn main() {
    a();
    a_reordered();
}
