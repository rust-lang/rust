//@ check-pass
//@ edition:2021

#![warn(non_local_definitions)]

struct Dog;

fn main() {
    impl PartialEq<()> for Dog {
    //~^ WARN non-local `impl` definition
        fn eq(&self, _: &()) -> bool {
            todo!()
        }
    }

    impl PartialEq<()> for &Dog {
    //~^ WARN non-local `impl` definition
        fn eq(&self, _: &()) -> bool {
            todo!()
        }
    }

    impl PartialEq<Dog> for () {
    //~^ WARN non-local `impl` definition
        fn eq(&self, _: &Dog) -> bool {
            todo!()
        }
    }

    impl PartialEq<&Dog> for () {
    //~^ WARN non-local `impl` definition
        fn eq(&self, _: &&Dog) -> bool {
            todo!()
        }
    }

    impl PartialEq<Dog> for &Dog {
    //~^ WARN non-local `impl` definition
        fn eq(&self, _: &Dog) -> bool {
            todo!()
        }
    }

    impl PartialEq<&Dog> for &Dog {
    //~^ WARN non-local `impl` definition
        fn eq(&self, _: &&Dog) -> bool {
            todo!()
        }
    }
}
