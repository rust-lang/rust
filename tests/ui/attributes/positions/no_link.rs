//@aux-build:dummy.rs

#![forbid(unstable_features)]
#![warn(unused_attributes)]

#![no_link] //~ ERROR

#[no_link] // OK
extern crate dummy;

extern crate core;

#[no_link] //~ ERROR
pub mod empty_crate;

#[no_link] //~ ERROR
pub mod module {

    #[no_link] //~ ERROR
    pub static GLOBAL: u32 = 42;

    #[no_link] //~ ERROR
    pub static mut GLOBAL_MUT: u32 = 42;

    #[no_link] //~ ERROR
    pub const CONST: u32 = 42;

    #[no_link] //~ ERROR
    use ::core::fmt::Debug;

    #[no_link] //~ ERROR
    trait TraitAlias = Debug; //~ ERROR trait aliases are experimental

    #[no_link] //~ ERROR
    pub type TypeAlias = u32;

    #[no_link] //~ ERROR
    pub struct Struct<
        #[no_link] 'lt, //~ ERROR
        #[no_link] T> //~ ERROR
     {
        #[no_link] //~ WARN
        pub struct_field: u32,
        #[no_link] //~ WARN
        pub generic_field: &'lt T,
    }

    #[no_link] //~ ERROR
    pub struct TupleStruct(#[no_link] pub u32); //~ WARN

    #[no_link] //~ ERROR
    pub enum Enum {
        #[no_link] //~ ERROR
        FieldVariant {
            #[no_link] //~ WARN
            field: u32,
        },
        #[no_link] //~ ERROR
        TupleVariant(#[no_link] u32), //~ WARN
    }
    #[no_link] //~ ERROR
    pub union Union {
        #[no_link] //~ WARN
        pub field: u32,
    }

    #[no_link] //~ ERROR
    impl<'lt, T> Struct<'lt, T> {
        #![no_link] //~ ERROR
        //~^ WARN unused attribute


        #[no_link] //~ ERROR
        pub fn method(#[no_link] &mut self) {  //~ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
            //~^ ERROR
            #[no_link] //~ ERROR
            //~^ ERROR
            self.struct_field = 73;
        }

        #[no_link] //~ ERROR
        pub fn static_method(#[no_link] _arg: u32) {} //~ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
        //~^ ERROR
    }

    #[no_link] //~ ERROR
    impl<
        #[no_link] 'lt, //~ ERROR
        #[no_link] T //~ ERROR
    > Drop for Struct<'lt, T> {
        #[no_link] //~ ERROR
        fn drop(&mut self) {
            // ..
        }
    }

    #[no_link]  //~ ERROR
    pub fn function<
        #[no_link] const N: usize //~ ERROR
    >(arg: u32) -> u32 {
        #![no_link] //~ ERROR
        //~^ WARN

        arg
    }

    #[no_link] //~ ERROR
    pub trait Trait {
        #[no_link] //~ ERROR
        fn trait_method() {}
    }

    fn x() {
        #[no_link] //~ ERROR
        const {
            // ..
        }
    }
}

pub mod inner_module {
    #![no_link] //~ ERROR

    fn x(){
        const {
            #![no_link] //~ ERROR
        }
    }
}

#[no_link] //~ ERROR
fn main() {
    let _closure = #[no_link] //~ ERROR attributes on expressions are experimental
    //~^ ERROR
    | #[no_link] _arg: u32| {}; //~ ERROR
    //~^ ERROR
    let _move_closure = #[no_link] //~ ERROR attributes on expressions are experimental
    //~^ ERROR
    move | #[no_link] _arg: u32| {}; //~ ERROR
    //~^ ERROR
    #[no_link] //~ ERROR
    {
        #![no_link] //~ ERROR
        //~^ WARN unused attribute


        #[no_link] //~ ERROR
        let variable = 42_u32;

        let _array = #[no_link] //~ ERROR attributes on expressions are experimental
        //~^ ERROR
        [
            #[no_link] //~ ERROR
            1,
            2,
        ];
        let _tuple = #[no_link] //~ ERROR attributes on expressions are experimental
        //~^ ERROR
        (
            #[no_link] //~ ERROR
            1,
            2,
        );
        let _tuple_struct = module::TupleStruct(
            #[no_link] //~ ERROR
            2,
        );
        let _struct = module::Struct {
            #[no_link] //~ ERROR
            struct_field: 42,
            generic_field: &13,
        };

        let _union = module::Union {
            #[no_link] //~ ERROR
            field: 42,
        };

        let _fn_call = module::function::<7>(
            #[no_link] //~ ERROR
            42,
        );

        #[no_link] //~ ERROR
        match variable {
            #[no_link] //~ WARN
            _match_arm => {}
        }

        let tail = ();

        #[no_link] //~ ERROR
        tail
    }

    {
        fn f(_: impl Fn()) {}
        // Is this attribute argument-level or expression-level?
        f(
            #[no_link] //~ ERROR
            || {},
        );
    }

    #[no_link] //~ ERROR
    unsafe {
        let _x = [1, 2, 3].get_unchecked(1);
    }
}

#[no_link] //~ ERROR
unsafe extern "C" {
    #![no_link] //~ ERROR
    //~^ WARN unused attribute


    #[no_link] //~ ERROR
    pub fn external_function(
        #[no_link] arg: *mut u8, //~ ERROR
        #[no_link] ... //~ ERROR
    );

    #[no_link] //~ ERROR
    pub static EXTERNAL_STATIC: *const u32;

    #[no_link] //~ ERROR
    pub static mut EXTERNAL_STATIC_MUT: *mut u32;
}

#[no_link] //~ ERROR
pub unsafe extern "C" fn abi_function(#[no_link] _: u32) {} //~ ERROR
//~^ ERROR
#[no_link] //~ WARN
#[macro_export]
macro_rules! my_macro {
    () => {};
}

#[no_link] // OK
#[test]
fn test_fn() {}
