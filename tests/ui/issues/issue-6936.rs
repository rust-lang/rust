struct T;

mod t1 {
    type Foo = ::T;
    mod Foo {} //~ ERROR the name `Foo` is defined multiple times
}

mod t2 {
    type Foo = ::T;
    struct Foo; //~ ERROR the name `Foo` is defined multiple times
}

mod t3 {
    type Foo = ::T;
    enum Foo {} //~ ERROR the name `Foo` is defined multiple times
}

mod t4 {
    type Foo = ::T;
    fn Foo() {} // ok
}

mod t5 {
    type Bar<T> = T;
    mod Bar {} //~ ERROR the name `Bar` is defined multiple times
}

mod t6 {
    type Foo = ::T;
    impl Foo {} // ok
}


fn main() {}
