//@aux-build:../../ui/auxiliary/proc_macros.rs
//@revisions: default crate_root allow_unused
//@[default] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/missing_docs_in_private_items/default
//@[crate_root] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/missing_docs_in_private_items/crate_root
//@[allow_unused] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/missing_docs_in_private_items/allow_unused

#![feature(decl_macro, trait_alias)]
#![deny(clippy::missing_docs_in_private_items)]
#![allow(non_local_definitions)]

extern crate proc_macros;
use proc_macros::{external, with_span};

fn main() {}

pub fn fn_pub() {}
pub const CONST_PUB: u32 = 0;
pub static STATIC_PUB: u32 = 0;
pub type TyAliasPub = u32;
pub trait TraitAliasPub = Iterator;
pub struct StructPub;
pub struct StructFieldPub {
    pub f1: u32,
    /// docs
    pub f2: u32,
    f3: u32, //~ missing_docs_in_private_items
    /// docs
    f4: u32,
    pub _f5: u32,
    /// docs
    pub _f6: u32,
    _f7: u32, //~[default,crate_root] missing_docs_in_private_items
    /// docs
    _f8: u32,
}
pub struct StructTuplePub(u32, pub u32);
pub enum EnumPub {
    V1,
    V2(u32),
    V3 {
        f1: u32,
        /// docs
        f2: u32,
    },
    /// docs
    V4,
    /// docs
    V5(u32),
    /// docs
    V6 {
        f1: u32,
        /// docs
        f2: u32,
    },
}
pub union UnionPub {
    pub f1: u32,
    /// docs
    pub f2: u32,
    f3: u32, //~ missing_docs_in_private_items
    /// docs
    f4: u32,
    pub _f5: u32,
    /// docs
    pub _f6: u32,
    _f7: u32, //~[default,crate_root] missing_docs_in_private_items
    /// docs
    _f8: u32,
}
impl StructFieldPub {
    pub fn f1() {}
    pub const C1: u32 = 0;
    /// docs
    pub fn f2() {}
    /// docs
    pub const C2: u32 = 0;
    fn f3() {} //~ missing_docs_in_private_items
    const C3: u32 = 0; //~ missing_docs_in_private_items
    /// docs
    fn f4() {}
    /// docs
    const C4: u32 = 0;
}
pub trait TraitPub {
    fn f1();
    fn f2() {}
    const C1: u32;
    const C2: u32 = 0;
    type T1;
    /// docs
    fn f3();
    /// docs
    fn f4() {}
    /// docs
    const C3: u32;
    /// docs
    const C4: u32 = 0;
    /// docs
    type T2;
}
impl TraitPub for StructPub {
    fn f1() {}
    const C1: u32 = 0;
    type T1 = u32;
    fn f3() {}
    const C3: u32 = 0;
    type T2 = u32;
}
#[macro_export]
macro_rules! mac_rules_pub {
    () => {};
}
pub macro mac_pub {
    () => {},
}

fn fn_crate() {} //~ missing_docs_in_private_items
const CONST_CRATE: u32 = 0; //~ missing_docs_in_private_items
static STATIC_CRATE: u32 = 0; //~ missing_docs_in_private_items
type TyAliasCrate = u32; //~ missing_docs_in_private_items
trait TraitAliasCrate = Iterator; //~ missing_docs_in_private_items
struct StructCrate; //~ missing_docs_in_private_items
//~v missing_docs_in_private_items
struct StructFieldCrate {
    pub f1: u32, //~ missing_docs_in_private_items
    /// docs
    pub f2: u32,
    f3: u32, //~ missing_docs_in_private_items
    /// docs
    f4: u32,
    pub _f5: u32, //~[default,crate_root] missing_docs_in_private_items
    /// docs
    pub _f6: u32,
    _f7: u32, //~[default,crate_root] missing_docs_in_private_items
    /// docs
    _f8: u32,
}
struct StructTupleCrate(u32, pub u32); //~ missing_docs_in_private_items
//~v missing_docs_in_private_items
enum EnumCrate {
    V1,      //~ missing_docs_in_private_items
    V2(u32), //~ missing_docs_in_private_items
    //~v missing_docs_in_private_items
    V3 {
        f1: u32, //~ missing_docs_in_private_items
        /// docs
        f2: u32,
    },
    /// docs
    V4,
    /// docs
    V5(u32),
    /// docs
    V6 {
        f1: u32, //~ missing_docs_in_private_items
        /// docs
        f2: u32,
    },
}
//~v missing_docs_in_private_items
union UnionCrate {
    pub f1: u32, //~ missing_docs_in_private_items
    /// docs
    pub f2: u32,
    f3: u32, //~ missing_docs_in_private_items
    /// docs
    f4: u32,
    pub _f5: u32, //~[default,crate_root] missing_docs_in_private_items
    /// docs
    pub _f6: u32,
    _f7: u32, //~[default,crate_root] missing_docs_in_private_items
    /// docs
    _f8: u32,
}
impl StructFieldCrate {
    pub fn f1() {} //~ missing_docs_in_private_items
    pub const C1: u32 = 0; //~ missing_docs_in_private_items
    /// docs
    pub fn f2() {}
    /// docs
    pub const C2: u32 = 0;
    fn f3() {} //~ missing_docs_in_private_items
    const C3: u32 = 0; //~ missing_docs_in_private_items
    /// docs
    fn f4() {}
    /// docs
    const C4: u32 = 0;
}
//~v missing_docs_in_private_items
trait TraitCrate {
    fn f1(); //~ missing_docs_in_private_items
    fn f2() {} //~ missing_docs_in_private_items
    const C1: u32; //~ missing_docs_in_private_items
    const C2: u32 = 0; //~ missing_docs_in_private_items
    type T1; //~ missing_docs_in_private_items
    /// docs
    fn f3();
    /// docs
    fn f4() {}
    /// docs
    const C3: u32;
    /// docs
    const C4: u32 = 0;
    /// docs
    type T2;
}
impl TraitCrate for StructCrate {
    fn f1() {}
    const C1: u32 = 0;
    type T1 = u32;
    fn f3() {}
    const C3: u32 = 0;
    type T2 = u32;
}
//~v missing_docs_in_private_items
macro_rules! mac_rules_crate {
    () => {};
}
//~v missing_docs_in_private_items
macro mac_crate {
    () => {},
}

/// docs
fn fn_crate_doc() {}
/// docs
const CONST_CRATE_DOC: u32 = 0;
/// docs
static STATIC_CRATE_DOC: u32 = 0;
/// docs
type TyAliasCrateDoc = u32;
/// docs
trait TraitAliasCrateDoc = Iterator;
/// docs
struct StructCrateDoc;
/// docs
struct StructFieldCrateDoc {
    pub f1: u32, //~ missing_docs_in_private_items
    /// docs
    pub f2: u32,
    f3: u32, //~ missing_docs_in_private_items
    /// docs
    f4: u32,
    pub _f5: u32, //~[default,crate_root] missing_docs_in_private_items
    /// docs
    pub _f6: u32,
    _f7: u32, //~[default,crate_root] missing_docs_in_private_items
    /// docs
    _f8: u32,
}
/// docs
struct StructTupleCrateDoc(u32, pub u32);
/// docs
enum EnumCrateDoc {
    V1,      //~ missing_docs_in_private_items
    V2(u32), //~ missing_docs_in_private_items
    //~v missing_docs_in_private_items
    V3 {
        f1: u32, //~ missing_docs_in_private_items
        /// docs
        f2: u32,
    },
    /// docs
    V4,
    /// docs
    V5(u32),
    /// docs
    V6 {
        f1: u32, //~ missing_docs_in_private_items
        /// docs
        f2: u32,
    },
}
/// docs
union UnionCrateDoc {
    pub f1: u32, //~ missing_docs_in_private_items
    /// docs
    pub f2: u32,
    f3: u32, //~ missing_docs_in_private_items
    /// docs
    f4: u32,
    pub _f5: u32, //~[default,crate_root] missing_docs_in_private_items
    /// docs
    pub _f6: u32,
    _f7: u32, //~[default,crate_root] missing_docs_in_private_items
    /// docs
    _f8: u32,
}
impl StructFieldCrateDoc {
    pub fn f1() {} //~ missing_docs_in_private_items
    pub const C1: u32 = 0; //~ missing_docs_in_private_items
    /// docs
    pub fn f2() {}
    /// docs
    pub const C2: u32 = 0;
    fn f3() {} //~ missing_docs_in_private_items
    const C3: u32 = 0; //~ missing_docs_in_private_items
    /// docs
    fn f4() {}
    /// docs
    const C4: u32 = 0;
}
/// docs
trait TraitCrateDoc {
    fn f1(); //~ missing_docs_in_private_items
    fn f2() {} //~ missing_docs_in_private_items
    const C1: u32; //~ missing_docs_in_private_items
    const C2: u32 = 0; //~ missing_docs_in_private_items
    type T1; //~ missing_docs_in_private_items
    /// docs
    fn f3();
    /// docs
    fn f4() {}
    /// docs
    const C3: u32;
    /// docs
    const C4: u32 = 0;
    /// docs
    type T2;
}
impl TraitCrate for StructCrateDoc {
    fn f1() {}
    const C1: u32 = 0;
    type T1 = u32;
    fn f3() {}
    const C3: u32 = 0;
    type T2 = u32;
}
/// docs
macro_rules! mac_rules_crate_doc {
    () => {};
}
/// docs
macro mac_crate_doc {
    () => {},
}

#[doc(hidden)]
fn fn_crate_hidden() {}
#[doc(hidden)]
const CONST_CRATE_HIDDEN: u32 = 0;
#[doc(hidden)]
static STATIC_CRATE_HIDDEN: u32 = 0;
#[doc(hidden)]
type TyAliasCrateHidden = u32;
#[doc(hidden)]
trait TraitAliasCrateHidden = Iterator;
#[doc(hidden)]
struct StructCrateHidden;
#[doc(hidden)]
struct StructFieldCrateHidden {
    pub f1: u32,
    /// docs
    pub f2: u32,
    f3: u32,
    /// docs
    f4: u32,
    pub _f5: u32,
    /// docs
    pub _f6: u32,
    _f7: u32,
    /// docs
    _f8: u32,
}
#[doc(hidden)]
struct StructTupleCrateHidden(u32, pub u32);
#[doc(hidden)]
enum EnumCrateHidden {
    V1,
    V2(u32),
    V3 {
        f1: u32,
        /// docs
        f2: u32,
    },
    V4,
    V5(u32),
    /// docs
    V6 {
        f1: u32,
        /// docs
        f2: u32,
    },
}
#[doc(hidden)]
union UnionCrateHidden {
    pub f1: u32,
    /// docs
    pub f2: u32,
    f3: u32,
    /// docs
    f4: u32,
    pub _f5: u32,
    /// docs
    pub _f6: u32,
    _f7: u32,
    /// docs
    _f8: u32,
}
#[doc(hidden)]
impl StructFieldCrateHidden {
    pub fn f1() {}
    pub const C1: u32 = 0;
    /// docs
    pub fn f2() {}
    /// docs
    pub const C2: u32 = 0;
    fn f3() {}
    const C3: u32 = 0;
    /// docs
    fn f4() {}
    /// docs
    const C4: u32 = 0;
}
#[doc(hidden)]
trait TraitCrateHidden {
    fn f1();
    fn f2() {}
    const C1: u32;
    const C2: u32 = 0;
    type T1;
    /// docs
    fn f3();
    /// docs
    fn f4() {}
    /// docs
    const C3: u32;
    /// docs
    const C4: u32 = 0;
    /// docs
    type T2;
}
#[doc(hidden)]
macro_rules! mac_rules_crate_hidden {
    () => {};
}
#[doc(hidden)]
macro mac_crate_hidden {
    () => {},
}

#[expect(clippy::missing_docs_in_private_items)]
fn fn_crate_expect() {}
#[expect(clippy::missing_docs_in_private_items)]
const CONST_CRATE_EXPECT: u32 = 0;
#[expect(clippy::missing_docs_in_private_items)]
static STATIC_CRATE_EXPECT: u32 = 0;
#[expect(clippy::missing_docs_in_private_items)]
type TyAliasCrateExpect = u32;
#[expect(clippy::missing_docs_in_private_items)]
trait TraitAliasCrateExpect = Iterator;
#[expect(clippy::missing_docs_in_private_items)]
struct StructCrateExpect;
#[expect(clippy::missing_docs_in_private_items)]
struct StructFieldCrateExpect {
    #[expect(clippy::missing_docs_in_private_items)]
    pub f1: u32,
    /// docs
    pub f2: u32,
    #[expect(clippy::missing_docs_in_private_items)]
    f3: u32,
    /// docs
    f4: u32,
}
#[expect(clippy::missing_docs_in_private_items)]
struct StructTupleCrateExpect(u32, pub u32);
#[expect(clippy::missing_docs_in_private_items)]
enum EnumCrateExpect {
    #[expect(clippy::missing_docs_in_private_items)]
    V1,
    #[expect(clippy::missing_docs_in_private_items)]
    V2(u32),
    #[expect(clippy::missing_docs_in_private_items)]
    V3 {
        #[expect(clippy::missing_docs_in_private_items)]
        f1: u32,
        /// docs
        f2: u32,
    },
    /// docs
    V4,
    /// docs
    V5(u32),
    /// docs
    V6 {
        #[expect(clippy::missing_docs_in_private_items)]
        f1: u32,
        /// docs
        f2: u32,
    },
}
#[expect(clippy::missing_docs_in_private_items)]
union UnionCrateExpect {
    #[expect(clippy::missing_docs_in_private_items)]
    pub f1: u32,
    /// docs
    pub f2: u32,
    #[expect(clippy::missing_docs_in_private_items)]
    f3: u32,
    /// docs
    f4: u32,
}
impl StructFieldCrateExpect {
    #[expect(clippy::missing_docs_in_private_items)]
    pub fn f1() {}
    #[expect(clippy::missing_docs_in_private_items)]
    pub const C1: u32 = 0;
    #[expect(clippy::missing_docs_in_private_items)]
    fn f2() {}
    #[expect(clippy::missing_docs_in_private_items)]
    const C2: u32 = 0;
}
#[expect(clippy::missing_docs_in_private_items)]
trait TraitCrateExpect {
    #[expect(clippy::missing_docs_in_private_items)]
    fn f1();
    #[expect(clippy::missing_docs_in_private_items)]
    fn f2() {}
    #[expect(clippy::missing_docs_in_private_items)]
    const C1: u32;
    #[expect(clippy::missing_docs_in_private_items)]
    const C2: u32 = 0;
    #[expect(clippy::missing_docs_in_private_items)]
    type T1;
}
#[expect(clippy::missing_docs_in_private_items)]
macro_rules! mac_rules_crate_expect {
    () => {};
}
#[expect(clippy::missing_docs_in_private_items)]
macro mac_crate_expect {
    () => {},
}

pub mod mod_pub {
    pub fn f1() {}
    pub struct S1 {
        pub f1: u32,
        /// docs
        pub f2: u32,
        f3: u32, //~[default,allow_unused] missing_docs_in_private_items
        /// docs
        f4: u32,
    }
    pub enum E1 {
        V1 {
            f1: u32,
            /// docs
            f2: u32,
        },
        /// docs
        V2 {
            f1: u32,
            /// docs
            f2: u32,
        },
    }
    pub const C1: u32 = 0;

    /// docs
    pub fn f2() {}
    /// docs
    pub struct S2 {
        pub f1: u32,
        /// docs
        pub f2: u32,
        f3: u32, //~[default,allow_unused] missing_docs_in_private_items
        /// docs
        f4: u32,
    }
    /// docs
    pub enum E2 {
        V1 {
            f1: u32,
            /// docs
            f2: u32,
        },
        /// docs
        V2 {
            f1: u32,
            /// docs
            f2: u32,
        },
    }
    /// docs
    pub const C2: u32 = 0;

    fn f3() {} //~[default,allow_unused] missing_docs_in_private_items
    //
    //~[default,allow_unused]v missing_docs_in_private_items
    struct S3 {
        pub f1: u32, //~[default,allow_unused] missing_docs_in_private_items
        /// docs
        pub f2: u32,
        f3: u32, //~[default,allow_unused] missing_docs_in_private_items
        /// docs
        f4: u32,
    }
    //~[default,allow_unused]v missing_docs_in_private_items
    enum E3 {
        //~[default,allow_unused]v missing_docs_in_private_items
        V1 {
            f1: u32, //~[default,allow_unused] missing_docs_in_private_items
            /// docs
            f2: u32,
        },
        /// docs
        V2 {
            f1: u32, //~[default,allow_unused] missing_docs_in_private_items
            /// docs
            f2: u32,
        },
    }
    const C3: u32 = 0; //~[default,allow_unused] missing_docs_in_private_items

    /// docs
    fn f4() {}
    /// docs
    struct S4 {
        pub f1: u32, //~[default,allow_unused] missing_docs_in_private_items
        /// docs
        pub f2: u32,
        f3: u32, //~[default,allow_unused] missing_docs_in_private_items
        /// docs
        f4: u32,
    }
    /// docs
    enum E4 {
        //~[default,allow_unused]v missing_docs_in_private_items
        V1 {
            f1: u32, //~[default,allow_unused] missing_docs_in_private_items
            /// docs
            f2: u32,
        },
        /// docs
        V2 {
            f1: u32, //~[default,allow_unused] missing_docs_in_private_items
            /// docs
            f2: u32,
        },
    }
    /// docs
    const C4: u32 = 0;
}

//~v missing_docs_in_private_items
mod mod_crate {
    pub fn f1() {} //~ missing_docs_in_private_items
    //
    //~v missing_docs_in_private_items
    pub struct S1 {
        pub f1: u32, //~ missing_docs_in_private_items
        /// docs
        pub f2: u32,
        f3: u32, //~[default,allow_unused] missing_docs_in_private_items
        /// docs
        f4: u32,
    }
    //~v missing_docs_in_private_items
    pub enum E1 {
        //~v missing_docs_in_private_items
        V1 {
            f1: u32, //~ missing_docs_in_private_items
            /// docs
            f2: u32,
        },
        /// docs
        V2 {
            f1: u32, //~ missing_docs_in_private_items
            /// docs
            f2: u32,
        },
    }
    pub const C1: u32 = 0; //~ missing_docs_in_private_items

    /// docs
    pub fn f2() {}
    /// docs
    pub struct S2 {
        pub f1: u32, //~ missing_docs_in_private_items
        /// docs
        pub f2: u32,
        f3: u32, //~[default,allow_unused] missing_docs_in_private_items
        /// docs
        f4: u32,
    }
    /// docs
    pub enum E2 {
        //~v missing_docs_in_private_items
        V1 {
            f1: u32, //~ missing_docs_in_private_items
            /// docs
            f2: u32,
        },
        /// docs
        V2 {
            f1: u32, //~ missing_docs_in_private_items
            /// docs
            f2: u32,
        },
    }
    /// docs
    pub const C2: u32 = 0;

    fn f3() {} //~[default,allow_unused] missing_docs_in_private_items
    //
    //~[default,allow_unused]v missing_docs_in_private_items
    struct S3 {
        pub f1: u32, //~[default,allow_unused] missing_docs_in_private_items
        /// docs
        pub f2: u32,
        f3: u32, //~[default,allow_unused] missing_docs_in_private_items
        /// docs
        f4: u32,
    }
    //~[default,allow_unused]v missing_docs_in_private_items
    enum E3 {
        //~[default,allow_unused]v missing_docs_in_private_items
        V1 {
            f1: u32, //~[default,allow_unused] missing_docs_in_private_items
            /// docs
            f2: u32,
        },
        /// docs
        V2 {
            f1: u32, //~[default,allow_unused] missing_docs_in_private_items
            /// docs
            f2: u32,
        },
    }
    const C3: u32 = 0; //~[default,allow_unused] missing_docs_in_private_items

    /// docs
    fn f4() {}
    /// docs
    struct S4 {
        pub f1: u32, //~[default,allow_unused] missing_docs_in_private_items
        /// docs
        pub f2: u32,
        f3: u32, //~[default,allow_unused] missing_docs_in_private_items
        /// docs
        f4: u32,
    }
    /// docs
    enum E4 {
        //~[default,allow_unused]v missing_docs_in_private_items
        V1 {
            f1: u32, //~[default,allow_unused] missing_docs_in_private_items
            /// docs
            f2: u32,
        },
        /// docs
        V2 {
            f1: u32, //~[default,allow_unused] missing_docs_in_private_items
            /// docs
            f2: u32,
        },
    }
    /// docs
    const C4: u32 = 0;
}

/// docs
mod mod_crate_doc {}

#[doc(hidden)]
mod mod_crate_hidden {
    pub fn f1() {}
    pub struct S1 {
        pub f1: u32,
        /// docs
        pub f2: u32,
        f3: u32,
        /// docs
        f4: u32,
    }
    pub enum E1 {
        V1 {
            f1: u32,
            /// docs
            f2: u32,
        },
        /// docs
        V2 {
            f1: u32,
            /// docs
            f2: u32,
        },
    }
    pub const C1: u32 = 0;

    /// docs
    pub fn f2() {}
    /// docs
    pub struct S2 {
        pub f1: u32,
        /// docs
        pub f2: u32,
        f3: u32,
        /// docs
        f4: u32,
    }
    /// docs
    pub enum E2 {
        V1 {
            f1: u32,
            /// docs
            f2: u32,
        },
        /// docs
        V2 {
            f1: u32,
            /// docs
            f2: u32,
        },
    }
    /// docs
    pub const C2: u32 = 0;

    fn f3() {}
    struct S3 {
        pub f1: u32,
        /// docs
        pub f2: u32,
        f3: u32,
        /// docs
        f4: u32,
    }
    enum E3 {
        V1 {
            f1: u32,
            /// docs
            f2: u32,
        },
        /// docs
        V2 {
            f1: u32,
            /// docs
            f2: u32,
        },
    }
    const C3: u32 = 0;

    /// docs
    fn f4() {}
    /// docs
    struct S4 {
        pub f1: u32,
        /// docs
        pub f2: u32,
        f3: u32,
        /// docs
        f4: u32,
    }
    /// docs
    enum E4 {
        V1 {
            f1: u32,
            /// docs
            f2: u32,
        },
        /// docs
        V2 {
            f1: u32,
            /// docs
            f2: u32,
        },
    }
    /// docs
    const C4: u32 = 0;
}

#[expect(clippy::missing_docs_in_private_items)]
mod mod_crate_expect {}

#[doc = "docs"]
mod explicit_doc_attr {}

with_span! {
    sp
    fn fn_pm() {}
    const CONST_PM: u32 = 0;
    static STATIC_PM: u32 = 0;
    type TyAliasPm = u32;
    trait TraitAliasPm = Iterator;
    struct StructPm;
    struct StructFieldPm {
        pub f1: u32,
        f2: u32,
        pub _f3: u32,
        _f4: u32,
    }
    struct StructTuplePm(u32, pub u32);
    enum EnumPm {
        V1,
        V2(u32),
        V3 { f1: u32, },
    }
    union UnionPm {
        pub f1: u32,
        f2: u32,
        pub _f3: u32,
        _f4: u32,
    }
    impl StructFieldPm {
        pub fn f1() {}
        pub const C1: u32 = 0;
        fn f2() {}
        const C2: u32 = 0;
    }
    trait TraitPm {
        fn f1();
        fn f2() {}
        const C1: u32;
        const C2: u32 = 0;
        type T1;
    }
    impl TraitPm for StructPm {
        fn f1() {}
        const C1: u32 = 0;
        type T1 = u32;
    }
    macro_rules! mac_rules_pm {
        () => {};
    }
    macro mac_pm {
        () => {},
    }
    mod mod_pm {}
}

external! {
    fn fn_external() {}
    const CONST_EXTERNAL: u32 = 0;
    static STATIC_EXTERNAL: u32 = 0;
    type TyAliasExternal = u32;
    trait TraitAliasExternal = Iterator;
    struct StructExternal;
    struct StructFieldExternal {
        pub f1: u32,
        f2: u32,
        pub _f3: u32,
        _f4: u32,
    }
    struct StructTupleExternal(u32, pub u32);
    enum EnumExternal {
        V1,
        V2(u32),
        V3 { f1: u32, },
    }
    union UnionExternal {
        pub f1: u32,
        f2: u32,
        pub _f3: u32,
        _f4: u32,
    }
    impl StructFieldExternal {
        pub fn f1() {}
        pub const C1: u32 = 0;
        fn f2() {}
        const C2: u32 = 0;
    }
    trait TraitExternal {
        fn f1();
        fn f2() {}
        const C1: u32;
        const C2: u32 = 0;
        type T1;
    }
    impl TraitExternal for StructExternal {
        fn f1() {}
        const C1: u32 = 0;
        type T1 = u32;
    }
    macro_rules! mac_rules_external {
        () => {};
    }
    macro mac_external {
        () => {},
    }
    mod mod_external {}
}

pub const _: () = {};
const _: () = {};

/// docs
fn fn_with_items() {
    fn f() {}
    type T = u32;
    struct S {
        f1: u32,
        f2: u32,
    }
    enum E {
        V { f: u32 },
    }
    impl S {
        fn f() {}
        const C: u32 = 0;
    }
    const C: u32 = 0;
    static ST: u32 = 0;
    trait Tr {
        fn f();
        type T;
        const C: u32;
    }
    trait Tr2 = Tr;
    mod m {}
    macro_rules! m2 {
        () => {};
    }
    macro m3 { () => {}, }
    union U {
        f: u32,
    }
}
/// docs
const CONST_WITH_ITEMS: () = {
    fn f() {}
};
/// docs
static STATIC_WITH_ITEMS: () = {
    fn f() {}
};
/// docs
trait TraitWithItems {
    /// docs
    fn f() {
        fn f() {}
    }
    /// docs
    const C: () = {
        fn f() {}
    };
}
/// docs
struct StructWithItems;
impl StructWithItems {
    /// docs
    fn f() {
        fn f() {}
    }
    /// docs
    const C: () = {
        fn f() {}
    };
}
/// docs
type TypeAliasWithItems = [u32; {
    fn f() {}
    1
}];

/// docs
mod with_reexports {
    pub fn f1_reexport() {}
    pub struct S1Reexport {
        pub f1: u32,
        f2: u32,            //~[default,allow_unused] missing_docs_in_private_items
        pub(crate) f3: u32, //~ missing_docs_in_private_items
        /// docs
        f4: u32,
    }

    /// docs
    mod m1 {
        pub(crate) fn f2() {} //~[default,allow_unused] missing_docs_in_private_items

        //~v missing_docs_in_private_items
        pub enum E1 {
            V1Reexport,
            V2, //~ missing_docs_in_private_items
        }

        pub struct S2; //~ missing_docs_in_private_items
        pub fn f3_reexport() -> S2 {
            S2
        }
    }
    pub use m1::E1::{V1Reexport, V2};
    use m1::f2;
    pub use m1::f3_reexport;
}
pub use with_reexports::{S1Reexport, V1Reexport, f1_reexport, f3_reexport};

external! {
    mod mod_generated {
        $(type T = u32;)
        struct S {
            $(f1: u32,)
            f2: u32,
        }
        pub fn f() {}
        $(pub fn f2() {}) //~ missing_docs_in_private_items
        #[doc(hidden)]
        $(pub fn f3() {})
    }
}

/// docs
mod mod_with_hidden {
    #[doc(hidden)]
    pub mod m {
        pub struct S {
            #[doc(hidden)]
            pub f: u32,
        }
        #[automatically_derived]
        impl S {
            #[doc(hidden)]
            pub fn f() {}
            pub const C: () = {
                #[automatically_derived]
                impl S {
                    #[doc(hidden)]
                    pub fn f2() {
                        mod m {
                            pub(crate) union U {
                                pub f: u32,
                            }
                        }
                    }
                }
            };
        }
    }
    #[doc(hidden)]
    pub(crate) fn f() {}
}

/// docs
struct WithProject {
    /// docs
    a: u32,
    /// docs
    b: u32,
}
with_span! {
    span
    const _: () = {
        // Similar output to pin_project
        struct Project<'a> {
            $(a: &'a u32),
            $(b: &'a u32),
        }
        impl $(WithProject) {
            fn project(&self) -> Project<'_> {
                Project {
                    a: &self.a,
                    b: &self.b,
                }
            }
        }
    };
}

external! {
    mod mod_mac_with_pub {$(
        struct DerivedFromInput;
        impl DerivedFromInput {
            pub fn foo() {}
        }
        pub struct VisFromOutside; //~ missing_docs_in_private_items
    )}
}
