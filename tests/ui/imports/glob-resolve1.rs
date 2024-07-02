// Make sure that globs only bring in public things.

use bar::*;

mod bar {
    use self::fpriv as import;
    fn fpriv() {}
    extern "C" {
        fn epriv();
    }
    enum A {
        A1,
    }
    pub enum B {
        B1,
    }

    struct C;

    type D = isize;
}

fn foo<T>() {}

fn main() {
    fpriv(); //~ ERROR cannot find function `fpriv`
    epriv(); //~ ERROR cannot find function `epriv`
    B; //~ ERROR expected value, found enum `B`
    C; //~ ERROR cannot find value `C`
    import(); //~ ERROR: cannot find function `import`

    foo::<A>(); //~ ERROR: cannot find type `A`
    foo::<C>(); //~ ERROR: cannot find type `C`
    foo::<D>(); //~ ERROR: cannot find type `D`
}

mod other {
    pub fn import() {}
}
