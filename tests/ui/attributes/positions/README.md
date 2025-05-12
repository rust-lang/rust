It's not uncommon to see comments like these when reading attribute related code:

```rust
// FIXME(#80564): We permit struct fields, match arms and macro defs to have an
// `#[no_mangle]` attribute with just a lint, because we previously
// erroneously allowed it and some crates used it accidentally, to be compatible
// with crates depending on them, we can't throw an error here.
```

or:

```rust
// FIXME: This can be used on stable but shouldn't.
```

Or perhaps you've seen discussions getting derailed by comments like:

```text
I discovered today that the following code already compiles on stable since at least Rust 1.31
...
```

This is largely because our coverage of attributes and where they're allowed is very poor.
This test suite attempts to consistently and exhaustively check attributes in various positions.
Hopefully, improving test coverage leads to the following:
- We become aware of the weird edge cases that are already allowed.
- We avoid unknowingly breaking those things.
- We can avoid accidentally stabilizing things we don't want stabilized.

If you know an attribute or a position that isn't tested but should be, please file an issue or PR.

## Template

`ATTRIBUTE.rs`
```rust
//@aux-build:dummy.rs

#![forbid(unstable_features)]
#![warn(unused_attributes)]

#![ATTRIBUTE] //~ WARN

#[ATTRIBUTE] //~ WARN
extern crate dummy;

extern crate core;

#[ATTRIBUTE] //~ WARN
pub mod empty_crate;

#[ATTRIBUTE] //~ WARN
pub mod module {

    #[ATTRIBUTE] //~ WARN
    pub static GLOBAL: u32 = 42;

    #[ATTRIBUTE] //~ WARN
    pub static mut GLOBAL_MUT: u32 = 42;

    #[ATTRIBUTE] //~ WARN
    pub const CONST: u32 = 42;

    #[ATTRIBUTE] //~ WARN
    use ::core::fmt::Debug;

    #[ATTRIBUTE] //~ WARN
    trait TraitAlias = Debug; //~ ERROR trait aliases are experimental

    #[ATTRIBUTE] //~ WARN
    pub type TypeAlias = u32;

    #[ATTRIBUTE] //~ WARN
    pub struct Struct<
        #[ATTRIBUTE] 'lt, //~ WARN
        #[ATTRIBUTE] T> //~ WARN
     {
        #[ATTRIBUTE] //~ WARN
        pub struct_field: u32,
        #[ATTRIBUTE] //~ WARN
        pub generic_field: &'lt T,
    }

    #[ATTRIBUTE] //~ WARN
    pub struct TupleStruct(#[ATTRIBUTE] pub u32); //~ WARN

    #[ATTRIBUTE] //~ WARN
    pub enum Enum {
        #[ATTRIBUTE] //~ WARN
        FieldVariant {
            #[ATTRIBUTE] //~ WARN
            field: u32,
        },
        #[ATTRIBUTE] //~ WARN
        TupleVariant(#[ATTRIBUTE] u32), //~ WARN
    }
    #[ATTRIBUTE] //~ WARN
    pub union Union {
        #[ATTRIBUTE] //~ WARN
        pub field: u32,
    }

    #[ATTRIBUTE] //~ WARN
    impl<'lt, T> Struct<'lt, T> {
        #![ATTRIBUTE] //~ WARN
        //~^ WARN unused attribute
        //~^^ WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

        #[ATTRIBUTE] //~ WARN
        pub fn method(#[ATTRIBUTE] &mut self) { //~ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
            //~^ WARN
            #[ATTRIBUTE] //~ ERROR
            //~^ WARN
            self.struct_field = struct_field;
        }

        #[ATTRIBUTE] //~ WARN
        pub fn static_method(#[ATTRIBUTE] _arg: u32) {} //~ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
    }

    #[ATTRIBUTE]
    impl<
        #[ATTRIBUTE] 'lt, //~ WARN
        #[ATTRIBUTE] T //~ WARN
    > Drop for Struct<'lt, T> {
        #[ATTRIBUTE] //~ WARN
        fn drop(&mut self) {
            // ..
        }
    }

    #[ATTRIBUTE]  //~ WARN
    pub fn function<
        #[ATTRIBUTE] const N: usize //~ WARN
    >(arg: u32) -> u32 {
        #![ATTRIBUTE] //~ WARN
        //~^ WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

        arg
    }

    #[ATTRIBUTE] //~ WARN
    pub trait Trait {
        #[ATTRIBUTE] //~ WARN
        fn trait_method() {}
    }

    fn x() {
        #[ATTRIBUTE] //~ WARN
        const {
            // ..
        }
    }
}

pub mod inner_module {
    #![ATTRIBUTE] //~ WARN

    fn x() {
        const {
            #![ATTRIBUTE] //~ WARN
        }
    }
}


#[ATTRIBUTE] //~ WARN
fn main() {
    let _closure = #[ATTRIBUTE] //~ ERROR attributes on expressions are experimental
    //~^ WARN
    |  #[ATTRIBUTE] _arg: u32| {}; //~ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
    //~^ ERROR
    let _move_closure = #[ATTRIBUTE] //~ ERROR attributes on expressions are experimental
    //~^ WARN
    move |  #[ATTRIBUTE] _arg: u32| {}; //~ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
    //~^ ERROR
    #[ATTRIBUTE] //~ WARN
    {
        #![ATTRIBUTE] //~ WARN
        //~^ WARN unused attribute
        //~^^ WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

        #[ATTRIBUTE] //~ WARN
        let variable = 42_u32;

        let _array = #[ATTRIBUTE] //~ ERROR attributes on expressions are experimental
        //~^ WARN
        [
            #[ATTRIBUTE] //~ WARN
            1,
            2,
        ];
        let _tuple = #[ATTRIBUTE] //~ ERROR attributes on expressions are experimental
        //~^ WARN
        (
            #[ATTRIBUTE] //~ WARN
            1,
            2,
        );
        let _tuple_struct = module::TupleStruct(
            #[ATTRIBUTE] //~ WARN
            2,
        );
        let _struct = module::Struct {
            #[ATTRIBUTE] //~ WARN
            struct_field: 42,
            generic_field: &13,
        };

        let _union = module::Union {
            #[ATTRIBUTE] //~ WARN
            field: 42,
        };

        let _fn_call = module::function::<7>(
            #[ATTRIBUTE] //~ WARN
            42,
        );

        #[ATTRIBUTE] //~ WARN
        match variable {
            #[ATTRIBUTE] //~ WARN
            _match_arm => {}
        }

        let tail = ();

        #[ATTRIBUTE] //~ WARN
        tail
    }

    {
        fn f(_: impl Fn()) {}
        // Is this attribute argument-level or expression-level?
        f(
            #[ATTRIBUTE] //~ WARN
            || {},
        );
    }

    #[ATTRIBUTE] //~ WARN
    unsafe {
        let _x = [1, 2, 3].get_unchecked(1);
    }
}

#[ATTRIBUTE] //~ WARN
unsafe extern "C" {
    #![ATTRIBUTE] //~ WARN
    //~^ WARN unused attribute
    //~^^ WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

    #[ATTRIBUTE] //~ WARN
    pub fn external_function(
        #[ATTRIBUTE] arg: *mut u8, //~ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
        #[ATTRIBUTE] ... //~ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
    );

    #[ATTRIBUTE] //~ WARN
    pub static EXTERNAL_STATIC: *const u32;

    #[ATTRIBUTE] //~ WARN
    pub static mut EXTERNAL_STATIC_MUT: *mut u32;
}

#[ATTRIBUTE] //~ WARN
pub unsafe extern "C" fn abi_function(#[ATTRIBUTE] _: u32) {} //~ ERROR allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters
//~^ ERROR
#[ATTRIBUTE] //~ WARN
#[macro_export]
macro_rules! my_macro {
    () => {};
}

#[ATTRIBUTE] //~ WARN
#[test]
fn test_fn() {}
```
