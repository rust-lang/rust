//! Defines a set of traits that is used for abstracting
//! rustc_public's components that are needed in rustc_public_bridge.
//!
//! These traits are really useful when programming
//! in rustc_public-agnostic settings.

use std::fmt::Debug;

use super::context::CompilerCtxt;
use super::{Bridge, Tables};

pub trait Error {
    fn new(msg: String) -> Self;
    fn from_internal<T: Debug>(err: T) -> Self;
}

pub trait Prov<B: Bridge> {
    fn new(aid: B::AllocId) -> Self;
}

pub trait Allocation<B: Bridge> {
    fn new<'tcx>(
        bytes: Vec<Option<u8>>,
        ptrs: Vec<(usize, rustc_middle::mir::interpret::AllocId)>,
        align: u64,
        mutability: rustc_middle::mir::Mutability,
        tables: &mut Tables<'tcx, B>,
        cx: &CompilerCtxt<'tcx, B>,
    ) -> Self;
}

macro_rules! make_bridge_trait {
    ($name:ident) => {
        pub trait $name<B: Bridge> {
            fn new(did: B::DefId) -> Self;
        }
    };
}

make_bridge_trait!(CrateItem);
make_bridge_trait!(AdtDef);
make_bridge_trait!(ForeignModuleDef);
make_bridge_trait!(ForeignDef);
make_bridge_trait!(FnDef);
make_bridge_trait!(ClosureDef);
make_bridge_trait!(CoroutineDef);
make_bridge_trait!(CoroutineClosureDef);
make_bridge_trait!(AliasDef);
make_bridge_trait!(ParamDef);
make_bridge_trait!(BrNamedDef);
make_bridge_trait!(TraitDef);
make_bridge_trait!(GenericDef);
make_bridge_trait!(ConstDef);
make_bridge_trait!(ImplDef);
make_bridge_trait!(RegionDef);
make_bridge_trait!(CoroutineWitnessDef);
make_bridge_trait!(AssocDef);
make_bridge_trait!(OpaqueDef);
make_bridge_trait!(StaticDef);
