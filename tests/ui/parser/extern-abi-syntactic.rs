//@ check-pass

// Check that from the grammar's point of view,
// the specific set of ABIs is not part of it.

fn main() {}

#[cfg(FALSE)]
extern "some_abi_that_we_are_sure_does_not_exist_semantically" fn foo() {}

#[cfg(FALSE)]
extern "some_abi_that_we_are_sure_does_not_exist_semantically" {
    fn foo();
}

#[cfg(FALSE)]
type T = extern "some_abi_that_we_are_sure_does_not_exist_semantically" fn();
