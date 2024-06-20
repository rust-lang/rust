#![feature(coverage_attribute)]
//@ edition: 2021

// Demonstrates the interaction of `#[coverage(off)]` and `#[coverage(on)]`
// in nested functions.

// FIXME(#126625): Coverage attributes should apply recursively to nested functions.
// FIXME(#126626): When an inner (non-closure) function has `#[coverage(off)]`,
// its lines can still be marked with misleading execution counts from its enclosing
// function.

#[coverage(off)]
fn do_stuff() {}

#[coverage(off)]
fn dense_a() {
    dense_b();
    dense_b();
    #[coverage(on)]
    fn dense_b() {
        dense_c();
        dense_c();
        #[coverage(off)]
        fn dense_c() {
            do_stuff();
        }
    }
}

#[coverage(off)]
fn sparse_a() {
    sparse_b();
    sparse_b();
    fn sparse_b() {
        sparse_c();
        sparse_c();
        #[coverage(on)]
        fn sparse_c() {
            sparse_d();
            sparse_d();
            fn sparse_d() {
                sparse_e();
                sparse_e();
                #[coverage(off)]
                fn sparse_e() {
                    do_stuff();
                }
            }
        }
    }
}

#[coverage(off)]
fn main() {
    dense_a();
    sparse_a();
}
