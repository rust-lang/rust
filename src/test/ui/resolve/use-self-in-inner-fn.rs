struct A;

impl A {
//~^ NOTE `Self` type implicitly declared here, by this `impl`
    fn banana(&mut self) {
        fn peach(this: &Self) {
        //~^ ERROR can't use generic parameters from outer function
        //~| NOTE use of generic parameter from outer function
        //~| NOTE use a type here instead
        }
    }
}

fn main() {}
