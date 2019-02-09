// run-pass

#![allow(unused)]

// Like other items, private imports can be imported and used non-lexically in paths.
mod a {
    use a as foo;
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
    use foo::*;
    use bar::*;
    fn f() -> bool { true }
    let _: bool = f();
}

fn h() {
    use foo::*;
    use bar::*;
    use f;
    let _: bool = f();
}

// Here, there appears to be shadowing but isn't because of namespaces.
mod b {
    // This imports `f` in the value namespace.
    use foo::*;
    // This imports `f` only in the type namespace,
    use super::b as f;
    // so the glob isn't shadowed.
    fn test() { self::f(); }
}

// Here, there is shadowing in one namespace, but not the other.
mod c {
    mod test {
        pub fn f() {}
        pub mod f {}
    }
    // This glob-imports `f` in both namespaces.
    use self::test::*;
    // This shadows the glob only in the value namespace.
    mod f { pub fn f() {} }

    fn test() {
        // Check that the glob-imported value isn't shadowed.
        self::f();
        // Check that the glob-imported module is shadowed.
        self::f::f();
    }
}

// Unused names can be ambiguous.
mod d {
    // This imports `f` in the value namespace.
    pub use foo::*;
    // This also imports `f` in the value namespace.
    pub use bar::*;
}

mod e {
    // N.B., since `e::f` is not used, this is not considered to be a use of `d::f`.
    pub use d::*;
}

fn main() {}
