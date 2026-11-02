//@aux-build:external_item.rs
//@aux-build:proc_macros.rs

#![warn(clippy::used_underscore_items)]
#![allow(clippy::no_effect)]

extern crate external_item;
extern crate proc_macros;

use proc_macros::{external, inline_macros};

#[inline_macros]
fn main() {
    {
        fn _f() {}
        const _C: u32 = 0;
        static _S: u32 = 0;
        struct _X;
        enum Z {
            _A,
        }

        struct X;
        impl X {
            fn _m(self) {}
        }

        _f; //~ used_underscore_items
        _f(); //~ used_underscore_items
        _C; //~ used_underscore_items
        _S; //~ used_underscore_items
        _X; //~ used_underscore_items
        _X {}; //~ used_underscore_items
        Z::_A; //~ used_underscore_items
        X::_m; //~ used_underscore_items
        X._m(); //~ used_underscore_items
    }
    // Non-underscore names.
    {
        fn f1() {}
        const C1: u32 = 0;
        static S1: u32 = 0;
        struct X1;
        enum Z1 {
            A1,
        }

        struct X;
        impl X {
            fn m1(self) {}
        }

        f1;
        f1();
        C1;
        S1;
        X1;
        X1 {};
        Z1::A1;
        X::m1;
        X.m1();
    }
    // Don't lint external items. The names may not be changeable.
    {
        let x = external_item::_ExternalStruct {};
        x._foo();
        external_item::_external_foo();
    }
    // Don't lint foreign functions. The names may not be changeable.
    {
        unsafe extern "C" {
            pub fn _exit(code: i32) -> !;
        }
        unsafe { _exit(1) }
    }
    // Don't lint in macros.
    {
        fn _f() {}
        const _C: u32 = 0;
        static _S: u32 = 0;
        struct _X;
        enum Z {
            _A,
        }

        inline! {
            _f();
            _C;
            _S;
            _X;
            _X;
            Z::_A;
        }
    }
    // Make sure expect works on the item.
    {
        #[expect(clippy::used_underscore_items)]
        fn _f() {}
        #[expect(clippy::used_underscore_items)]
        const _C: u32 = 0;
        #[expect(clippy::used_underscore_items)]
        static _S: u32 = 0;
        #[expect(clippy::used_underscore_items)]
        struct _X;
        enum Z {
            #[expect(clippy::used_underscore_items)]
            _A,
        }

        struct X;
        impl X {
            #[expect(clippy::used_underscore_items)]
            fn _m(self) {}
        }

        _f;
        _f();
        _C;
        _S;
        _X;
        _X {};
        Z::_A;
        X::_m;
        X._m();
    }
    // Ignore anything automatically derived.
    {
        struct S;
        #[automatically_derived]
        impl S {
            fn f() {
                fn _f() {}
                const _C: u32 = 0;
                static _S: u32 = 0;
                struct _X;
                enum Z {
                    _A,
                }

                _f();
                _C;
                _S;
                _X;
                _X {};
                Z::_A;
            }
        }
    }
}
