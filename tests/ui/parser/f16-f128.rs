// Make sure we don't ICE while incrementally adding f16 and f128 support

mod f16_checks {
    const A: f16 = 10.0; //~ ERROR cannot find type `f16` in this scope

    pub fn main() {
        let a: f16 = 100.0; //~ ERROR cannot find type `f16` in this scope
        let b = 0.0f16; //~ ERROR invalid width `16` for float literal

        foo(1.23);
    }

    fn foo(a: f16) {} //~ ERROR cannot find type `f16` in this scope

    struct Bar {
        a: f16, //~ ERROR cannot find type `f16` in this scope
    }
}

mod f128_checks {
    const A: f128 = 10.0; //~ ERROR cannot find type `f128` in this scope

    pub fn main() {
        let a: f128 = 100.0; //~ ERROR cannot find type `f128` in this scope
        let b = 0.0f128; //~ ERROR invalid width `128` for float literal

        foo(1.23);
    }

    fn foo(a: f128) {} //~ ERROR cannot find type `f128` in this scope

    struct Bar {
        a: f128, //~ ERROR cannot find type `f128` in this scope
    }
}

fn main() {}
