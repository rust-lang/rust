struct A;

impl A {
//~^ NOTE `Self` type implicitly declared here, by this `impl`
    fn banana(&mut self) {
        fn peach(this: &Self) {
        //~^ ERROR can't use generic parameters from outer item
        //~| NOTE use of generic parameter from outer item
        //~| NOTE refer to the type directly here instead
        }
    }
}

fn main() {}
