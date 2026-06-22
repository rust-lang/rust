#![deny(self_type_conversion)]
mod a {
    mod foo {
        cfg_select! {
            any(target_arch = "avr", target_arch = "msp430") => {
                pub type Int = i16;
            }
            _ => {
                pub type Int = i32;
            }
        }
        #[cfg(false)]
        pub type Float = f32;
        #[cfg(true)]
        pub type Float = f64;

        pub type Trigger = u64;
    }

    use self::foo::{Int, Float, Trigger};

    pub fn bar() {
        let x: Int = 1;
        // Ok, should not lint because the type alias is behind a `cfg_select!`
        let _: i32 = x.into();
        let y: Float = 1.;
        // Ok, should not lint because the type alias is behind a `cfg` attr
        let _: f64 = y.into();
        let z: Trigger = 1;
        let _: u64 = z.into();
        //~^ ERROR useless conversion to the same type: `u64`
    }
}

mod b {
    #[cfg(any(target_arch = "avr", target_arch = "msp430"))]
    mod foo {
        pub type Int = i16;
        pub type Float = f32;
    }
    #[cfg(not(any(target_arch = "avr", target_arch = "msp430")))]
    mod bar {
        pub type Int = i32;
        pub type Float = f64;
    }

    mod qux {
        pub type Trigger = u64;
    }

    cfg_select! {
        any(target_arch = "avr", target_arch = "msp430") => {
            use self::foo::Int;
        }
        _ => {
            use self::bar::Int;
        }
    }

    #[cfg(any(target_arch = "avr", target_arch = "msp430"))]
    use self::foo::Float;
    #[cfg(not(any(target_arch = "avr", target_arch = "msp430")))]
    use self::bar::Float;

    use self::qux::Trigger;

    pub fn baz() {
        let x: Int = 1;
        let _: i32 = x.into(); // Ok, should not lint because the import is behind a `cfg_select!`
        let y: Float = 1.;
        let _: f64 = y.into(); // Ok, should not lint because the import is behind a `cfg` attr
        let z: Trigger = 1;
        let _: u64 = z.into();
        //~^ ERROR useless conversion to the same type: `u64`
    }
}

struct C {
    #[cfg(true)]
    foo: (),
}
#[cfg(true)]
struct D {
    foo: (),
}
#[cfg(true)]
mod e {
    pub(crate) struct F {
        pub(crate) foo: (),
    }
    pub(crate) const G: () = ();
}
#[cfg(true)]
const H: () = ();


fn main() {
    a::bar();
    b::baz();
    let c = C { foo: ().into() }; // Ok, field `C.foo` is behind a `cfg` attr
    let () = c.foo.into(); // Ok, field `C.foo` is behind a `cfg` attr
    let d = D { foo: ().into() }; // Ok, `D` is behind a `cfg` attr
    let () = d.foo.into(); // Ok, `D` is behind a `cfg` attr
    let f = e::F { foo: ().into() }; // Ok, `e` is behind a `cfg` attr
    let () = f.foo.into(); // Ok, `e` is behind a `cfg` attr
    let () = e::G.into(); // Ok, `e` is behind a `cfg` attr
    let () = H.into(); // Ok, `H` is behind a `cfg` attr
}
