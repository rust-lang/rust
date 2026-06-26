pub trait Trait<T> {
    type Assoc;
}

fn f<U: Trait<i32> + Trait<u32>>() {
    let _: Assoc = todo!(); //~ ERROR cannot find type `Assoc` in this scope
}

pub trait Foo<'a> {
    type A;
}

pub mod inner {
    pub trait Foo<'a> {
        type A;
    }
}

fn g<'a, T: ::Foo<'a> + inner::Foo<'a>>() {
    let _: A = todo!(); //~ ERROR cannot find type `A` in this scope
}

pub trait First {
    type Assoc;
}

pub trait Second {
    type Assoc;
}

fn h<T: First<Assoc = u32> + Second<Assoc = i32>>() {
    let _: Assoc = todo!(); //~ ERROR cannot find type `Assoc` in this scope
}

pub trait Gat {
    type Assoc<'a>;
}

fn i<T: Gat>() {
    let _: Assoc = todo!(); //~ ERROR cannot find type `Assoc` in this scope
}

fn j<T: First>() {
    struct Local;
    impl Local {
        fn method<U: First>() {
            let _: Assoc = todo!(); //~ ERROR cannot find type `Assoc` in this scope
        }
    }

    let _ = std::marker::PhantomData::<T>;
}

pub struct S;
impl S {
    fn method<T: First>() {
        fn inner() {
            let _: Assoc = todo!(); //~ ERROR cannot find type `Assoc` in this scope
        }
        inner();
    }
}

fn main() {}
