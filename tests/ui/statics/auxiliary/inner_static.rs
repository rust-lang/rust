//! Test for inner statics with the same name.
//!
//! Before, the path name for all items defined in methods of traits and impls never
//! took into account the name of the method. This meant that if you had two statics
//! of the same name in two different methods the statics would end up having the
//! same symbol named (even after mangling) because the path components leading to
//! the symbol were exactly the same (just __extensions__ and the static name).
//!
//! It turns out that if you add the symbol "A" twice to LLVM, it automatically
//! makes the second one "A1" instead of "A". What this meant is that in local crate
//! compilations we never found this bug. Even across crates, this was never a
//! problem. The problem arises when you have generic methods that don't get
//! generated at compile-time of a library. If the statics were re-added to LLVM by
//! a client crate of a library in a different order, you would reference different
//! constants (the integer suffixes wouldn't be guaranteed to be the same).

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
