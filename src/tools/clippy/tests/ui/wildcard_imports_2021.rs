//@revisions: edition2018 edition2021
//@[edition2018] edition:2018
//@[edition2021] edition:2021
//@run-rustfix
//@aux-build:wildcard_imports_helper.rs

#![warn(clippy::wildcard_imports)]
#![allow(unused, clippy::unnecessary_wraps, clippy::let_unit_value)]
#![warn(unused_imports)]

extern crate wildcard_imports_helper;

use crate::fn_mod::*;
use crate::mod_mod::*;
use crate::multi_fn_mod::*;
use crate::struct_mod::*;

#[allow(unused_imports)]
use wildcard_imports_helper::inner::inner_for_self_import::*;
use wildcard_imports_helper::prelude::v1::*;
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
    let _ = PreludeModAnywhere;

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
        #[rustfmt::skip]
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

mod super_imports {
    fn foofoo() {}

    mod should_be_replaced {
        use super::*;

        fn with_super() {
            let _ = foofoo();
        }
    }

    mod test_should_pass {
        use super::*;

        fn with_super() {
            let _ = foofoo();
        }
    }

    mod test_should_pass_inside_function {
        fn with_super_inside_function() {
            use super::*;
            let _ = foofoo();
        }
    }

    mod test_should_pass_further_inside {
        fn insidefoo() {}
        mod inner {
            use super::*;
            fn with_super() {
                let _ = insidefoo();
            }
        }
    }

    mod should_be_replaced_further_inside {
        fn insidefoo() {}
        mod inner {
            use super::*;
            fn with_super() {
                let _ = insidefoo();
            }
        }
    }

    mod use_explicit_should_be_replaced {
        use crate::super_imports::*;

        fn with_explicit() {
            let _ = foofoo();
        }
    }

    mod use_double_super_should_be_replaced {
        mod inner {
            use super::super::*;

            fn with_double_super() {
                let _ = foofoo();
            }
        }
    }

    mod use_super_explicit_should_be_replaced {
        use super::super::super_imports::*;

        fn with_super_explicit() {
            let _ = foofoo();
        }
    }

    mod attestation_should_be_replaced {
        use super::*;

        fn with_explicit() {
            let _ = foofoo();
        }
    }
}
