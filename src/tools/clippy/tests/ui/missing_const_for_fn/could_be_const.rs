#![warn(clippy::missing_const_for_fn)]
#![allow(incomplete_features, clippy::let_and_return, clippy::missing_transmute_annotations)]
#![feature(const_trait_impl, abi_vectorcall)]


use std::mem::transmute;

struct Game {
    guess: i32,
}

impl Game {
    // Could be const
    pub fn new() -> Self {
        //~^ ERROR: this could be a `const fn`
        //~| NOTE: `-D clippy::missing-const-for-fn` implied by `-D warnings`
        Self { guess: 42 }
    }

    fn const_generic_params<'a, T, const N: usize>(&self, b: &'a [T; N]) -> &'a [T; N] {
        //~^ ERROR: this could be a `const fn`
        b
    }
}

// Could be const
fn one() -> i32 {
    //~^ ERROR: this could be a `const fn`
    1
}

// Could also be const
fn two() -> i32 {
    //~^ ERROR: this could be a `const fn`
    let abc = 2;
    abc
}

// Could be const (since Rust 1.39)
fn string() -> String {
    //~^ ERROR: this could be a `const fn`
    String::new()
}

// Could be const
unsafe fn four() -> i32 {
    //~^ ERROR: this could be a `const fn`
    4
}

// Could also be const
fn generic<T>(t: T) -> T {
    //~^ ERROR: this could be a `const fn`
    t
}

fn sub(x: u32) -> usize {
    unsafe { transmute(&x) }
}

fn generic_arr<T: Copy>(t: [T; 1]) -> T {
    //~^ ERROR: this could be a `const fn`
    t[0]
}

mod with_drop {
    pub struct A;
    pub struct B;
    impl Drop for A {
        fn drop(&mut self) {}
    }

    impl B {
        // This can be const, because `a` is passed by reference
        pub fn b(self, a: &A) -> B {
            //~^ ERROR: this could be a `const fn`
            B
        }
    }
}

#[clippy::msrv = "1.47.0"]
mod const_fn_stabilized_before_msrv {
    // This could be const because `u8::is_ascii_digit` is a stable const function in 1.47.
    fn const_fn_stabilized_before_msrv(byte: u8) {
        //~^ ERROR: this could be a `const fn`
        byte.is_ascii_digit();
    }
}

#[clippy::msrv = "1.45"]
fn msrv_1_45() -> i32 {
    45
}

#[clippy::msrv = "1.46"]
fn msrv_1_46() -> i32 {
    //~^ ERROR: this could be a `const fn`
    46
}

// Should not be const
fn main() {}

struct D;

/* FIXME(effects)
impl const Drop for D {
    fn drop(&mut self) {
        todo!();
    }
}
*/

// Lint this, since it can be dropped in const contexts
// FIXME(effects)
fn d(this: D) {}
//~^ ERROR: this could be a `const fn`

mod msrv {
    struct Foo(*const u8, &'static u8);

    impl Foo {
        #[clippy::msrv = "1.58"]
        fn deref_ptr_can_be_const(self) -> usize {
            //~^ ERROR: this could be a `const fn`
            unsafe { *self.0 as usize }
        }

        fn deref_copied_val(self) -> usize {
            //~^ ERROR: this could be a `const fn`
            *self.1 as usize
        }
    }

    union Bar {
        val: u8,
    }

    #[clippy::msrv = "1.56"]
    fn union_access_can_be_const() {
        //~^ ERROR: this could be a `const fn`
        let bar = Bar { val: 1 };
        let _ = unsafe { bar.val };
    }

    #[clippy::msrv = "1.62"]
    mod with_extern {
        extern "C" fn c() {}
        //~^ ERROR: this could be a `const fn`

        #[rustfmt::skip]
        extern fn implicit_c() {}
        //~^ ERROR: this could be a `const fn`

        // any item functions in extern block won't trigger this lint
        extern "C" {
            fn c_in_block();
        }
    }
}

mod issue12677 {
    pub struct Wrapper {
        pub strings: Vec<String>,
    }

    impl Wrapper {
        #[must_use]
        pub fn new(strings: Vec<String>) -> Self {
            Self { strings }
        }

        #[must_use]
        pub fn empty() -> Self {
            Self { strings: Vec::new() }
        }
    }

    pub struct Other {
        pub text: String,
        pub vec: Vec<String>,
    }

    impl Other {
        pub fn new(text: String) -> Self {
            let vec = Vec::new();
            Self { text, vec }
        }
    }
}

mod with_ty_alias {
    trait FooTrait {
        type Foo: std::fmt::Debug;
        fn bar(_: Self::Foo) {}
    }
    impl FooTrait for () {
        type Foo = i32;
    }
    // NOTE: When checking the type of a function param, make sure it is not an alias with
    // `AliasTyKind::Projection` before calling `TyCtxt::type_of` to find out what the actual type
    // is. Because the associate ty could have no default, therefore would cause ICE, as demonstrated
    // in this test.
    fn alias_ty_is_projection(bar: <() as FooTrait>::Foo) {}
}

mod extern_fn {
    extern "C-unwind" fn c_unwind() {}
    //~^ ERROR: this could be a `const fn`
    extern "system" fn system() {}
    //~^ ERROR: this could be a `const fn`
    extern "system-unwind" fn system_unwind() {}
    //~^ ERROR: this could be a `const fn`
    pub extern "vectorcall" fn std_call() {}
    //~^ ERROR: this could be a `const fn`
    pub extern "vectorcall-unwind" fn std_call_unwind() {}
    //~^ ERROR: this could be a `const fn`
}
