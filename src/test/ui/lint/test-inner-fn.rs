// compile-flags: --test -D unnameable_test_functions

#[test]
fn foo() {
    #[test] //~ ERROR cannot test inner function [unnameable_test_functions]
    fn bar() {}
    bar();
}

mod x {
    #[test]
    fn foo() {
        #[test] //~ ERROR cannot test inner function [unnameable_test_functions]
        fn bar() {}
        bar();
    }
}

fn main() {}
