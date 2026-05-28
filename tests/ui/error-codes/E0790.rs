mod inner {
    pub trait MyTrait {
        const MY_ASSOC_CONST: ();

        fn my_fn();
    }

    pub struct MyStruct;

    impl MyTrait for MyStruct {
        const MY_ASSOC_CONST: () = ();

        fn my_fn() {}
    }

    fn call() {
        MyTrait::my_fn(); //~ ERROR E0790
    }

    fn use_const() {
        let _ = MyTrait::MY_ASSOC_CONST; //~ ERROR E0790
    }
}

fn call_inner() {
    inner::MyTrait::my_fn(); //~ ERROR E0790
}

fn use_const_inner() {
    let _ = inner::MyTrait::MY_ASSOC_CONST; //~ ERROR E0790
}

trait MyTrait2 {
    fn my_fn();
}

struct Impl1;

impl MyTrait2 for Impl1 {
    fn my_fn() {}
}

struct Impl2;

impl MyTrait2 for Impl2 {
    fn my_fn() {}
}

fn call_multiple_impls() {
    MyTrait2::my_fn(); //~ ERROR E0790
}

fn main() {}
