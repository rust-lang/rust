struct A;

impl A {
//~^ NOTE `Self` type implicitly declared here, by this `impl`
    fn banana(&mut self) {
        fn peach(this: &Self) {
        //~^ ERROR can't use `Self` from outer item
        //~| NOTE use of `Self` from outer item
        //~| NOTE `Self` used in this inner function
        //~| HELP refer to the type directly here instead
        //~| NOTE nested items are independent from their
        }
    }
}

enum MyEnum {}

impl MyEnum {
//~^ NOTE `Self` type implicitly declared here, by this `impl`
    fn do_something(result: impl FnOnce()) {
        result();
    }

    fn do_something_extra() {
        fn inner() {
        //~^ NOTE `Self` used in this inner function
            Self::do_something(move || {});
            //~^ ERROR can't use `Self` from outer item
            //~| NOTE use of `Self` from outer item
            //~| HELP refer to the type directly here instead
            //~| NOTE nested items are independent from their
            MyEnum::do_something(move || {});
        }
        inner();
    }
}

fn main() {}
