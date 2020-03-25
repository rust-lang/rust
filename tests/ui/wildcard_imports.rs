// run-rustfix
// aux-build:wildcard_imports_helper.rs

#![warn(clippy::wildcard_imports)]
//#![allow(clippy::redundant_pub_crate)]
#![allow(unused)]
#![warn(unused_imports)]

extern crate wildcard_imports_helper;

use crate::fn_mod::*;
use crate::mod_mod::*;
use crate::multi_fn_mod::*;
#[macro_use]
use crate::struct_mod::*;

#[allow(unused_imports)]
use wildcard_imports_helper::inner::inner_for_self_import;
use wildcard_imports_helper::inner::inner_for_self_import::*;
use wildcard_imports_helper::*;

use std::io::prelude::*;

struct ReadFoo;

impl Read for ReadFoo {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        Ok(0)
    }
}

mod fn_mod {
    pub fn foo() {}
}

mod mod_mod {
    pub mod inner_mod {
        pub fn foo() {}
    }
}

mod multi_fn_mod {
    pub fn multi_foo() {}
    pub fn multi_bar() {}
    pub fn multi_baz() {}
    pub mod multi_inner_mod {
        pub fn foo() {}
    }
}

mod struct_mod {
    pub struct A;
    pub struct B;
    pub mod inner_struct_mod {
        pub struct C;
    }

    #[macro_export]
    macro_rules! double_struct_import_test {
        () => {
            let _ = A;
        };
    }
}

fn main() {
    foo();
    multi_foo();
    multi_bar();
    multi_inner_mod::foo();
    inner_mod::foo();
    extern_foo();
    inner_extern_bar();

    let _ = A;
    let _ = inner_struct_mod::C;
    let _ = ExternA;

    double_struct_import_test!();
    double_struct_import_test!();
}

mod in_fn_test {
    pub use self::inner_exported::*;
    #[allow(unused_imports)]
    pub(crate) use self::inner_exported2::*;

    fn test_intern() {
        use crate::fn_mod::*;

        foo();
    }

    fn test_extern() {
        use wildcard_imports_helper::inner::inner_for_self_import::{self, *};
        use wildcard_imports_helper::*;

        inner_for_self_import::inner_extern_foo();
        inner_extern_foo();

        extern_foo();

        let _ = ExternA;
    }

    fn test_inner_nested() {
        use self::{inner::*, inner2::*};

        inner_foo();
        inner_bar();
    }

    fn test_extern_reexported() {
        use wildcard_imports_helper::*;

        extern_exported();
        let _ = ExternExportedStruct;
        let _ = ExternExportedEnum::A;
    }

    mod inner_exported {
        pub fn exported() {}
        pub struct ExportedStruct;
        pub enum ExportedEnum {
            A,
        }
    }

    mod inner_exported2 {
        pub(crate) fn exported2() {}
    }

    mod inner {
        pub fn inner_foo() {}
    }

    mod inner2 {
        pub fn inner_bar() {}
    }
}

fn test_reexported() {
    use crate::in_fn_test::*;

    exported();
    let _ = ExportedStruct;
    let _ = ExportedEnum::A;
}

#[rustfmt::skip]
fn test_weird_formatting() {
    use crate:: in_fn_test::  * ;
    use crate:: fn_mod::
        *;

    exported();
    foo();
}
