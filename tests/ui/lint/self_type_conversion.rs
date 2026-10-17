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

        let x: Option<Int> = Some(1);
        // Ok, should not lint because the type alias is behind a `cfg_select!`
        let _: Option<i32> = x.map(Into::into);
        let y: Option<Float> = Some(1.);
        // Ok, should not lint because the type alias is behind a `cfg` attr
        let _: Option<f64> = y.map(Into::into);
        let z: Option<Trigger> = Some(1);
        let _: Option<u64> = z.map(Into::into);
        //~^ ERROR useless conversion to the same type: `u64`

        let x: Result<Int, ()> = Ok(1);
        // Ok, should not lint because the type alias is behind a `cfg_select!`
        let _: Result<i32, ()> = x.map(Into::into);
        let y: Result<Float, ()> = Ok(1.);
        // Ok, should not lint because the type alias is behind a `cfg` attr
        let _: Result<f64, ()> = y.map(Into::into);
        let z: Result<Trigger, ()> = Ok(1);
        let _: Result<u64, ()> = z.map(Into::into);
        //~^ ERROR useless conversion to the same type: `u64`

        let x: Result<(), Int> = Err(1);
        // Ok, should not lint because the type alias is behind a `cfg_select!`
        let _: Result<(), i32> = x.map_err(Into::into);
        let y: Result<(), Float> = Err(1.);
        // Ok, should not lint because the type alias is behind a `cfg` attr
        let _: Result<(), f64> = y.map_err(Into::into);
        let z: Result<(), Trigger> = Err(1);
        let _: Result<(), u64> = z.map_err(Into::into);
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

        let x: Option<Int> = Some(1);
        // Ok, should not lint because the type alias is behind a `cfg_select!`
        let _: Option<i32> = x.map(Into::into);
        let y: Option<Float> = Some(1.);
        // Ok, should not lint because the type alias is behind a `cfg` attr
        let _: Option<f64> = y.map(Into::into);
        let z: Option<Trigger> = Some(1);
        let _: Option<u64> = z.map(Into::into);
        //~^ ERROR useless conversion to the same type: `u64`

        let x: Result<Int, ()> = Ok(1);
        // Ok, should not lint because the type alias is behind a `cfg_select!`
        let _: Result<i32, ()> = x.map(Into::into);
        let y: Result<Float, ()> = Ok(1.);
        // Ok, should not lint because the type alias is behind a `cfg` attr
        let _: Result<f64, ()> = y.map(Into::into);
        let z: Result<Trigger, ()> = Ok(1);
        let _: Result<u64, ()> = z.map(Into::into);
        //~^ ERROR useless conversion to the same type: `u64`

        let x: Result<(), Int> = Err(1);
        // Ok, should not lint because the type alias is behind a `cfg_select!`
        let _: Result<(), i32> = x.map_err(Into::into);
        let y: Result<(), Float> = Err(1.);
        // Ok, should not lint because the type alias is behind a `cfg` attr
        let _: Result<(), f64> = y.map_err(Into::into);
        let z: Result<(), Trigger> = Err(1);
        let _: Result<(), u64> = z.map_err(Into::into);
        //~^ ERROR useless conversion to the same type: `u64`
    }
}
mod c {
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

        let x: Option<Int> = Some(1);
        // Ok, should not lint because the type alias is behind a `cfg_select!`
        let _: Option<i32> = x.map(Into::into);
        let y: Option<Float> = Some(1.);
        // Ok, should not lint because the type alias is behind a `cfg` attr
        let _: Option<f64> = y.map(Into::into);
        let z: Option<Trigger> = Some(1);
        let _: Option<u64> = z.map(Into::into);
        //~^ ERROR useless conversion to the same type: `u64`

        let x: Result<Int, ()> = Ok(1);
        // Ok, should not lint because the type alias is behind a `cfg_select!`
        let _: Result<i32, ()> = x.map(Into::into);
        let y: Result<Float, ()> = Ok(1.);
        // Ok, should not lint because the type alias is behind a `cfg` attr
        let _: Result<f64, ()> = y.map(Into::into);
        let z: Result<Trigger, ()> = Ok(1);
        let _: Result<u64, ()> = z.map(Into::into);
        //~^ ERROR useless conversion to the same type: `u64`

        let x: Result<(), Int> = Err(1);
        // Ok, should not lint because the type alias is behind a `cfg_select!`
        let _: Result<(), i32> = x.map_err(Into::into);
        let y: Result<(), Float> = Err(1.);
        // Ok, should not lint because the type alias is behind a `cfg` attr
        let _: Result<(), f64> = y.map_err(Into::into);
        let z: Result<(), Trigger> = Err(1);
        let _: Result<(), u64> = z.map_err(Into::into);
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

trait T {
    type Assoc: Into<u64>;

    fn foo(&self) -> Self::Assoc;
}

impl T for () {
    type Assoc = u64;
    fn foo(&self) -> u64 { 0 }
}

struct I;

fn question_mark() -> Result<I, I> {
    // Ok, `?` expands to an `.into()` call, but we shouldn't complain about that
    Ok(Err(I)?)
}

macro_rules! in_macro {
    ($expr:expr) => {
        $expr.into()
    }
}
macro_rules! in_macro_2 {
    ($expr:expr) => {
        $expr.into()
        //~^ ERROR useless conversion to the same type: `()`
        //~| ERROR useless conversion to the same type: `u16`
    }
}
macro_rules! in_macro_3 {
    ($expr1:expr, $expr2:expr) => {
        // FIXME: should likely complain about `$expr1` only, but now it doesn't complain at all.
        ($expr1.into(), $expr2.into())
    }
}
macro_rules! in_macro_4 {
    ($expr1:expr, $expr2:expr) => {
        ($expr1.into(), $expr2.into())
        //~^ ERROR useless conversion to the same type: `()`
        //~| ERROR useless conversion to the same type: `()`
        //~| ERROR useless conversion to the same type: `()`
        //~| ERROR useless conversion to the same type: `u16`
    }
}
macro_rules! in_macro_5 {
    ($expr1:expr, $expr2:expr) => {
        ($expr1.into(), $expr2.into())
    }
}

macro_rules! multiple_arms_ok {
    ($($ty:ty: $expr:expr,)*) => {
        $(
            let _: $ty = $expr.into();
        )*
    }
}

macro_rules! multiple_arms {
    ($($ty:ty: $expr:expr,)*) => {
        $(
            let _: $ty = $expr.into();
            //~^ ERROR useless conversion to the same type: `()`
            //~| ERROR useless conversion to the same type: `u64`
        )*
    }
}

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
    let _ = ().foo(); // Ok, `T::Assoc` has an `Into<_>` bound
    let () = in_macro!(()); // Ok, there's another macro expansion where the `.into()` isn't useless
    let _: u64 = in_macro!(0u16); // Ok, the expanded `.into()` isn't entirelyl uselses
    let () = in_macro_2!(());  // Not ok, pointing into the macro
    let _: u16 = in_macro_2!(0u16); // Not ok, pointing into the macro
    let _: ((), ()) = in_macro_3!((), ()); // Should complain about the first argument, but doesn't
    let _: ((), u64) = in_macro_3!((), 0u16); // because *one* of the `.into()` wasn't useless.
    let _: ((), ()) = in_macro_4!((), ()); // Not ok, because both into calls are useless
    let _: ((), u16) = in_macro_4!((), 0u16); // Not ok, because both into calls are useless
    let _: ((), ()) = in_macro_5!((), ()); // Ok, because the next call is not useless
    let _: (u64, u64) = in_macro_5!(0u16, 0u16); // Ok
    multiple_arms_ok! {
        (): (), // Ok because of the next line
        u64: 0u16,
    }
    multiple_arms! {
        (): (),
        u64: 0u64,
    }
}
