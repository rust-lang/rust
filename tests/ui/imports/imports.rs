//@ run-pass
#![allow(unused)]

// Like other items, private imports can be imported and used non-lexically in paths.
mod a {
    use crate::a as foo;
    use self::foo::foo as bar;

    mod b {
        use super::bar;
    }
}

mod foo { pub fn f() {} }
mod bar { pub fn f() {} }

pub fn f() -> bool { true }

// Items and explicit imports shadow globs.
fn g() {
    use crate::foo::*;
    use crate::bar::*;
    fn f() -> bool { true }
    let _: bool = f();
}

fn h() {
    use crate::foo::*;
    use crate::bar::*;
    use crate::f;
    let _: bool = f();
}

// Here, there appears to be shadowing but isn't because of namespaces.
mod b {
    use crate::foo::*; // This imports `f` in the value namespace.
    use super::b as f; // This imports `f` only in the type namespace,
    fn test() { self::f(); } // so the glob isn't shadowed.
}

// Here, there is shadowing in one namespace, but not the other.
mod c {
    mod test {
        pub fn f() {}
        pub mod f {}
    }
    use self::test::*; // This glob-imports `f` in both namespaces.
    mod f { pub fn f() {} } // This shadows the glob only in the value namespace.

    fn test() {
        self::f(); // Check that the glob-imported value isn't shadowed.
        self::f::f(); // Check that the glob-imported module is shadowed.
    }
}

// Unused names can be ambiguous.
mod d {
    pub use crate::foo::*; // This imports `f` in the value namespace.
    pub use crate::bar::*; // This also imports `f` in the value namespace.
}

mod e {
    pub use crate::d::*; // n.b. Since `e::f` is not used,
                         // this is not considered to be a use of `d::f`.
}

fn main() {}
