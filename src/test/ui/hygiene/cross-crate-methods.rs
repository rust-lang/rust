// Test that methods defined in another crate are resolved correctly their
// names differ only in `SyntaxContext`. This also checks that any name
// resolution done when monomorphizing is correct.

// run-pass
// aux-build:methods.rs

extern crate methods;

use methods::*;

struct A;
struct B;
struct C;

impl MyTrait for A {}
test_trait!(impl for B);
test_trait2!(impl for C);

fn main() {
    check_crate_local();
    check_crate_local_generic(A, B);
    check_crate_local_generic(A, C);

    test_trait!(check_resolutions);
    test_trait2!(check_resolutions);
    test_trait!(assert_no_override A);
    test_trait2!(assert_no_override A);
    test_trait!(assert_override B);
    test_trait2!(assert_override B);
    test_trait!(assert_override C);
    test_trait2!(assert_override C);
}
