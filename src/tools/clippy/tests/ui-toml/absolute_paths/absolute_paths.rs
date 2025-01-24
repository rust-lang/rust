//@aux-build:../../ui/auxiliary/proc_macros.rs
//@revisions: default allow_crates allow_long no_short
//@[default] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/absolute_paths/default
//@[allow_crates] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/absolute_paths/allow_crates
//@[allow_long] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/absolute_paths/allow_long
//@[no_short] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/absolute_paths/no_short
#![deny(clippy::absolute_paths)]

extern crate proc_macros;
use proc_macros::{external, inline_macros, with_span};

#[inline_macros]
fn main() {
    let _ = std::path::is_separator(' ');
    //~[default]^ absolute_paths
    //~[allow_crates]| absolute_paths
    //~[no_short]| absolute_paths

    // Make sure this is treated as having three path segments, not four.
    let _ = ::std::path::MAIN_SEPARATOR;
    //~[default]^ absolute_paths
    //~[allow_crates]| absolute_paths
    //~[no_short]| absolute_paths

    let _ = std::collections::hash_map::HashMap::<i32, i32>::new(); //~ absolute_paths

    // Note `std::path::Path::new` is treated as having three parts
    let _: &std::path::Path = std::path::Path::new("");
    //~[default]^ absolute_paths
    //~[default]| absolute_paths
    //~[allow_crates]| absolute_paths
    //~[allow_crates]| absolute_paths
    //~[no_short]| absolute_paths
    //~[no_short]| absolute_paths

    // Treated as having three parts.
    let _ = ::core::clone::Clone::clone(&0i32);
    //~[default]^ absolute_paths
    //~[no_short]| absolute_paths
    let _ = <i32 as core::clone::Clone>::clone(&0i32);
    //~[default]^ absolute_paths
    //~[no_short]| absolute_paths
    let _ = std::option::Option::None::<i32>;
    //~[default]^ absolute_paths
    //~[allow_crates]| absolute_paths
    //~[no_short]| absolute_paths

    {
        // FIXME: macro calls should be checked.
        let x = 1i32;
        let _ = core::ptr::addr_of!(x);
    }

    {
        // FIXME: derive macro paths should be checked.
        #[derive(core::clone::Clone)]
        struct S;
    }

    {
        use core::fmt;
        use core::marker::PhantomData;

        struct X<T>(PhantomData<T>);
        impl<T: core::cmp::Eq> core::fmt::Display for X<T>
        //~[default]^ absolute_paths
        //~[default]| absolute_paths
        //~[no_short]| absolute_paths
        //~[no_short]| absolute_paths
        where T: core::clone::Clone
        //~[no_short]^ absolute_paths
        //~[default]| absolute_paths
        {
            fn fmt(&self, _: &mut fmt::Formatter) -> fmt::Result {
                Ok(())
            }
        }
    }

    {
        mod m1 {
            pub(crate) mod m2 {
                pub(crate) const FOO: i32 = 0;
            }
        }
        let _ = m1::m2::FOO;
    }

    with_span! {
        span
        let _ = std::path::is_separator(' ');
    }

    external! {
        let _ = std::path::is_separator(' ');
    }

    inline! {
        let _ = std::path::is_separator(' ');
    }
}

pub use core::cmp::Ordering;
pub use std::fs::File;

#[derive(Clone)]
pub struct S;
mod m1 {
    pub use crate::S;
}

//~[no_short]v absolute_paths
pub const _: crate::S = {
    let crate::S = m1::S; //~[no_short] absolute_paths

    crate::m1::S
    //~[default]^ absolute_paths
    //~[no_short]| absolute_paths
};

pub fn f() {
    let _ = <crate::S as Clone>::clone(&m1::S); //~[no_short] absolute_paths
}
