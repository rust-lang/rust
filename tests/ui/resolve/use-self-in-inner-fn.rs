struct A;

impl A {
//~^ NOTE `Self` type implicitly declared here, by this `impl`
    fn banana(&mut self) {
        fn peach(this: &Self) {
        //~^ ERROR can't use `Self` from outer item
        //~| NOTE use of `Self` from outer item
        //~| NOTE refer to the type directly here instead
        }
    }
}

fn main() {}
