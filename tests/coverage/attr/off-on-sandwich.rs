#![feature(coverage_attribute)]
//@ edition: 2021
//@ reference: attributes.coverage.nesting

// Demonstrates the interaction of `#[coverage(off)]` and `#[coverage(on)]`
// in nested functions.

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
