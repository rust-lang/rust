//@aux-build:dummy.rs

#![forbid(unstable_features)]
#![warn(unused_attributes)]

#![must_use] //~ WARN

#[must_use] //~ WARN
extern crate dummy;

extern crate core;

#[must_use] //~ WARN
pub mod empty_crate;

#[must_use] //~ WARN
pub mod module {

    #[must_use] //~ WARN
    pub static GLOBAL: u32 = 42;

    #[must_use] //~ WARN
    pub static mut GLOBAL_MUT: u32 = 42;

    #[must_use] //~ WARN
    pub const CONST: u32 = 42;

    #[must_use] //~ WARN
    use ::core::fmt::Debug;

    #[must_use] //~ WARN
    trait TraitAlias = Debug; //~ ERROR trait aliases are experimental

    #[must_use] //~ WARN
    pub type TypeAlias = u32;

    #[must_use] // OK
    pub struct Struct<
        #[must_use] 'lt, //~ WARN
        #[must_use] T> //~ WARN
     {
        #[must_use] //~ WARN
        pub struct_field: u32,
        #[must_use] //~ WARN
        pub generic_field: &'lt T,
    }

    #[must_use] // OK
    pub struct TupleStruct(#[must_use] pub u32); //~ WARN

    #[must_use] // OK
    pub enum Enum {
        #[must_use] //~ WARN
        FieldVariant {
            #[must_use] //~ WARN
            field: u32,
        },
        #[must_use] //~ WARN
        TupleVariant(#[must_use] u32), //~ WARN
    }
    #[must_use] // OK
    pub union Union {
        #[must_use] //~ WARN
        pub field: u32,
    }

    #[must_use] //~ WARN
    impl<'lt, T> Struct<'lt, T> {

        #[must_use] // OK
        pub fn method(#[must_use] &mut self) { //~ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
            //~^ WARN
            #[must_use] //~ ERROR
            //~^ WARN
            self.struct_field = 73;
        }

        #[must_use] // OK
        pub fn static_method(#[must_use] _arg: u32) {} //~ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
        //~^ WARN
    }

    #[must_use] //~ WARN
    impl<
        #[must_use] 'lt, //~ WARN
        #[must_use] T //~ WARN
    > Drop for Struct<'lt, T> {
        #[must_use] //~ WARN
        fn drop(&mut self) {
            // ..
        }
    }

    #[must_use] // OK
    pub fn function<
        #[must_use] const N: usize //~ WARN
    >(arg: u32) -> u32 {
        #![must_use] //~ WARN
        //~^ WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

        arg
    }

    #[must_use] // OK
    pub trait Trait {
        #[must_use] // OK
        fn trait_method() {}
    }

    fn x() {
        #[must_use] //~ WARN
        const {
            // ..
        }
    }
}

pub mod inner_module {
    #![must_use] //~ WARN

    impl<'lt, T> super::module::Struct<'lt, T> {
        #![must_use] //~ WARN

        pub fn inner_method(&mut self) {
            #![must_use] // OK
        }
    }

    fn x() {
        const {
            #![must_use] //~ WARN
        }
    }
}

#[must_use] // OK
fn main() {
    let _closure = #[must_use] //~ ERROR attributes on expressions are experimental
    //~^ WARN
    | #[must_use] _arg: u32| {}; //~ ERROR
    //~^ WARN
    let _move_closure = #[must_use] //~ ERROR attributes on expressions are experimental
    //~^ WARN
    move | #[must_use] _arg: u32| {}; //~ ERROR
    //~^ WARN
    #[must_use] //~ WARN
    {
        #![must_use] //~ WARN
        //~^ WARN unused attribute
        //~^^ WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

        #[must_use] //~ WARN
        let variable = 42_u32;

        let _array = #[must_use] //~ ERROR attributes on expressions are experimental
        //~^ WARN
        [
            #[must_use] //~ WARN
            1,
            2,
        ];
        let _tuple = #[must_use] //~ ERROR attributes on expressions are experimental
        //~^ WARN
        (
            #[must_use] //~ WARN
            1,
            2,
        );
        let _tuple_struct = module::TupleStruct(
            #[must_use] //~ WARN
            2,
        );
        let _struct = module::Struct {
            #[must_use] //~ WARN
            struct_field: 42,
            generic_field: &13,
        };

        let _union = module::Union {
            #[must_use] //~ WARN
            field: 42,
        };

        let _fn_call = module::function::<7>(
            #[must_use] //~ WARN
            42,
        );

        #[must_use] //~ WARN
        match variable {
            #[must_use] //~ WARN
            _match_arm => {}
        }

        let tail = ();

        #[must_use] //~ WARN
        tail
    }

    {
        fn f(_: impl Fn()) {}
        // Is this attribute argument-level or expression-level?
        f(
            #[must_use] //~ WARN
            || {},
        );
    }

    #[must_use] //~ WARN
    unsafe {
        let _x = [1, 2, 3].get_unchecked(1);
    }
}

#[must_use] //~ WARN
unsafe extern "C" {
    #![must_use] //~ WARN
    //~^ WARN unused attribute
    //~^^ WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

    #[must_use] // OK
    pub fn external_function(
        #[must_use] arg: *mut u8, //~ ERROR
        #[must_use] ... //~ ERROR
    );

    #[must_use] //~ WARN
    pub static EXTERNAL_STATIC: *const u32;

    #[must_use] //~ WARN
    pub static mut EXTERNAL_STATIC_MUT: *mut u32;
}

#[must_use] // OK
pub unsafe extern "C" fn abi_function(#[must_use] _: u32) {} //~ ERROR
//~^ WARN
#[must_use] //~ WARN
#[macro_export]
macro_rules! my_macro {
    () => {};
}

#[must_use] // OK
#[test]
fn test_fn() {}
