core::test_binder_constraints! {
    //~^ ERROR use of unstable library feature
    impl<'a: 'b, 'b> {
        'a: 'b
    }
}

fn main() {}
