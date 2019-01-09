pub struct A<T> { pub v: T }
pub struct B<T> { pub v: T }

pub mod test {
    pub struct A<T> { pub v: T }

    impl<T> A<T> {
        pub fn foo(&self) -> isize {
            static a: isize = 5;
            return a
        }

        pub fn bar(&self) -> isize {
            static a: isize = 6;
            return a;
        }
    }
}

impl<T> A<T> {
    pub fn foo(&self) -> isize {
        static a: isize = 1;
        return a
    }

    pub fn bar(&self) -> isize {
        static a: isize = 2;
        return a;
    }
}

impl<T> B<T> {
    pub fn foo(&self) -> isize {
        static a: isize = 3;
        return a
    }

    pub fn bar(&self) -> isize {
        static a: isize = 4;
        return a;
    }
}

pub fn foo() -> isize {
    let a = A { v: () };
    let b = B { v: () };
    let c = test::A { v: () };
    return a.foo() + a.bar() +
           b.foo() + b.bar() +
           c.foo() + c.bar();
}
