//@ compile-flags: --test -D unnameable_test_items

#[test]
fn foo() {
    #[test] //~ ERROR cannot test inner items [unnameable_test_items]
    fn bar() {}
    bar();
}

mod x {
    #[test]
    fn foo() {
        #[test] //~ ERROR cannot test inner items [unnameable_test_items]
        fn bar() {}
        bar();
    }
}

fn main() {}
