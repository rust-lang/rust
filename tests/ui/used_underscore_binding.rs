//@aux-build:proc_macros.rs

#![warn(clippy::used_underscore_binding)]
#![expect(clippy::explicit_auto_deref, clippy::no_effect)]

extern crate proc_macros;

use core::marker::PhantomData;
use proc_macros::{external, inline_macros};

#[inline_macros]
fn main() {
    // Declaration only.
    {
        let _a = 0;
    }
    // Check various reads.
    {
        let _a = 0;
        let _b = &0;
        let _c = (0, &0);
        let _d = String::new();

        _a; //~ used_underscore_binding
        *_b; //~ used_underscore_binding
        _c.0; //~ used_underscore_binding
        *_c.1; //~ used_underscore_binding
        _d.is_empty(); //~ used_underscore_binding
    }
    // Check that we match rustc on what a use is.
    {
        let mut _a = 0;
        let mut _b = (0, 0);
        let mut c = 0;
        let mut _c = &mut c;
        let mut d = (0, 0);
        let _d = &mut d;

        _a = 0;
        _b.0 = 0;
        _c = &mut c;
        *_c = 0; //~ used_underscore_binding
        _d.0 = 0; //~ used_underscore_binding
        (*_d).0 = 0; //~ used_underscore_binding
    }
    // Check field access.
    {
        struct X<'a> {
            _x: &'a mut (u32, u32),
        };

        let mut a = (0, 0);
        let mut b = (0, 0);
        let mut c = X { _x: &mut a };

        c._x; //~ used_underscore_binding
        c._x = &mut b; //~ used_underscore_binding
        *c._x; //~ used_underscore_binding
        *c._x = (0, 0); //~ used_underscore_binding
        (*c._x).0 = 0; //~ used_underscore_binding
        c._x.0 = 0; //~ used_underscore_binding
    }
    // Await desugaring contains a used underscore binding.
    {
        async fn f1() {}
        async fn f2() {
            f1().await;
            {
                let _a = 0;
                _a; //~ used_underscore_binding
                f1()
            }
            .await;
        }
    }
    // Ignore phantom fields
    {
        struct X {
            _marker: PhantomData<u32>,
        }
        let a = X { _marker: PhantomData };
        a._marker;
    }
    // Ignore multiple underscores
    {
        struct X {
            __x: u32,
        }
        let __a = 0;
        let b = X { __x: 0 };
        __a;
        b.__x;
    }
    // Check compound assignment
    {
        let mut _a = 0;
        let mut _b = (0, 0);
        let mut _c = 0.0;
        let mut _d = String::new();

        _a += 0;
        _b.0 -= 0;
        _b.1 |= 0;
        _c *= 1.0;
        _d += ""; //~ used_underscore_binding
    }
    // Expect on the binding
    {
        struct X {
            #[expect(clippy::used_underscore_binding)]
            _x: u32,
        }
        #[expect(clippy::used_underscore_binding)]
        let _a = 0;
        let b = X { _x: 0 };

        _a;
        b._x;
    }
    // Check macros
    {
        struct X {
            _x: u32,
        };

        let _a = 0;
        let _b = (0, 0);
        let mut _c = 0;
        let mut _d = (0, 0);
        let e = X { _x: 0 };
        let mut f = X { _x: 0 };

        inline!({
            $(@expr _a); //~ used_underscore_binding
            $(@expr _b.0); //~ used_underscore_binding
            $(@expr _c) = 0;
            $(@expr _d.0) = 0;
            $(@expr e._x); //~ used_underscore_binding
            $(@expr f._x) = 0; //~ used_underscore_binding
        });
        external!({
            $_a; //~ used_underscore_binding
            $(_b.0); //~ used_underscore_binding
            $_c = 0;
            $(_d.0) = 0;
            $(e._x); //~ used_underscore_binding
            $(f._x) = 0; //~ used_underscore_binding
        });
    }
}
