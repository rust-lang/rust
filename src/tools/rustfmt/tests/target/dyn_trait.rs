#![feature(dyn_trait)]

fn main() {
    // #2506
    // checks rustfmt doesn't remove dyn
    trait MyTrait {
        fn method(&self) -> u64;
    }
    fn f1(a: Box<dyn MyTrait>) {}

    // checks if line wrap works correctly
    trait Very_______________________Long__________________Name_______________________________Trait
    {
        fn method(&self) -> u64;
    }

    fn f2(
        a: Box<
            dyn Very_______________________Long__________________Name____________________Trait
                + 'static,
        >,
    ) {
    }

    // #2582
    let _: &dyn (::std::any::Any) = &msg;
}
