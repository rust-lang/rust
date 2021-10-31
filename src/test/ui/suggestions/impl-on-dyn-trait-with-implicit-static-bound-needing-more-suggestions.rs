// FIXME: the following cases need to suggest more things to make users reach a working end state.

mod bav {
    trait OtherTrait<'a> {}
    impl<'a> OtherTrait<'a> for &'a () {}

    trait ObjectTrait {
        type Assoc: Bar;
    }
    trait MyTrait {
        fn use_self(&self) -> &() { panic!() }
    }
    trait Bar {}

    impl MyTrait for Box<dyn ObjectTrait<Assoc = i32>> {
        fn use_self(&self) -> &() { panic!() }
    }
    impl Bar for i32 {}

    fn use_it<'a>(val: Box<dyn ObjectTrait<Assoc = i32>>) -> impl OtherTrait<'a> {
        val.use_self() //~ ERROR cannot return reference to function parameter
    }
}

mod bap {
    trait OtherTrait<'a> {}
    impl<'a> OtherTrait<'a> for &'a () {}

    trait ObjectTrait {
        type Assoc: Bar;
    }
    trait MyTrait {
        fn use_self(&self) -> &() { panic!() }
    }
    trait Bar {}

    impl MyTrait for Box<dyn ObjectTrait<Assoc = i32>> {
        fn use_self(&self) -> &() { panic!() }
    }
    impl Bar for i32 {}

    fn use_it<'a>(val: Box<dyn ObjectTrait<Assoc = i32>>) -> impl OtherTrait<'a> + 'a {
        val.use_self() //~ ERROR E0515
    }
}

// This case in particular requires the user to write all of the bounds we have in `mod bax`.
mod bay {
    trait OtherTrait<'a> {}
    impl<'a> OtherTrait<'a> for &'a () {}

    trait ObjectTrait {
        type Assoc: Bar;
    }
    trait MyTrait {
        fn use_self(&self) -> &() { panic!() }
    }
    trait Bar {}

    impl MyTrait for Box<dyn ObjectTrait<Assoc = i32>> {
        fn use_self(&self) -> &() { panic!() }
    }
    impl Bar for i32 {}

    fn use_it<'a>(val: Box<dyn ObjectTrait<Assoc = i32> + 'a>) -> &'a () {
        val.use_self() //~ ERROR E0772
    }
}

mod bax {
    trait OtherTrait<'a> {}
    impl<'a> OtherTrait<'a> for &'a () {}

    trait ObjectTrait {
        type Assoc: Bar;
    }
    trait MyTrait<'a> {
        fn use_self(&'a self) -> &'a () { panic!() }
    }
    trait Bar {}

    impl<'a> MyTrait<'a> for Box<dyn ObjectTrait<Assoc = i32> + 'a> {
        fn use_self(&'a self) -> &'a () { panic!() }
    }
    impl Bar for i32 {}

    fn use_it<'a>(val: Box<dyn ObjectTrait<Assoc = i32> + 'a>) -> &'a () {
        val.use_self()
    }
}

mod baw {
    trait OtherTrait<'a> {}
    impl<'a> OtherTrait<'a> for &'a () {}

    trait ObjectTrait {
        type Assoc: Bar;
    }
    trait MyTrait {
        fn use_self(&self) -> &() { panic!() }
    }
    trait Bar {}

    impl<'a> MyTrait for Box<dyn ObjectTrait<Assoc = Box<dyn Bar>>> {
        fn use_self(&self) -> &() { panic!() }
    }

    fn use_it<'a>(val: Box<dyn ObjectTrait<Assoc = Box<dyn Bar>>>) -> impl OtherTrait<'a> + 'a{
        val.use_self() //~ ERROR E0515
    }
}

fn main() {}
