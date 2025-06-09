struct T;

mod t1 {
    type Foo = crate::T;
    mod Foo {} //~ ERROR the name `Foo` is defined multiple times
}

mod t2 {
    type Foo = crate::T;
    struct Foo; //~ ERROR the name `Foo` is defined multiple times
}

mod t3 {
    type Foo = crate::T;
    enum Foo {} //~ ERROR the name `Foo` is defined multiple times
}

mod t4 {
    type Foo = crate::T;
    fn Foo() {} // ok
}

mod t5 {
    type Bar<T> = T;
    mod Bar {} //~ ERROR the name `Bar` is defined multiple times
}

mod t6 {
    type Foo = crate::T;
    impl Foo {} // ok
}


fn main() {}
