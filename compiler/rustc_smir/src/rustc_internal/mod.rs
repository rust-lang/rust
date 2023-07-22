//! Module that implements the bridge between Stable MIR and internal compiler MIR.
//!
//! For that, we define APIs that will temporarily be public to 3P that exposes rustc internal APIs
//! until stable MIR is complete.

use std::fmt::Debug;
use std::string::ToString;

use crate::{
    rustc_smir::Tables,
    stable_mir::{self, with},
};
use rustc_middle::ty::TyCtxt;
pub use rustc_span::def_id::{CrateNum, DefId};

fn with_tables<R>(mut f: impl FnMut(&mut Tables<'_>) -> R) -> R {
    let mut ret = None;
    with(|tables| tables.rustc_tables(&mut |t| ret = Some(f(t))));
    ret.unwrap()
}

pub fn item_def_id(item: &stable_mir::CrateItem) -> DefId {
    with_tables(|t| t.item_def_id(item))
}

pub fn crate_item(did: DefId) -> stable_mir::CrateItem {
    with_tables(|t| t.crate_item(did))
}

pub fn adt_def(did: DefId) -> stable_mir::ty::AdtDef {
    with_tables(|t| t.adt_def(did))
}

pub fn foreign_def(did: DefId) -> stable_mir::ty::ForeignDef {
    with_tables(|t| t.foreign_def(did))
}

pub fn fn_def(did: DefId) -> stable_mir::ty::FnDef {
    with_tables(|t| t.fn_def(did))
}

pub fn closure_def(did: DefId) -> stable_mir::ty::ClosureDef {
    with_tables(|t| t.closure_def(did))
}

pub fn generator_def(did: DefId) -> stable_mir::ty::GeneratorDef {
    with_tables(|t| t.generator_def(did))
}

pub fn alias_def(did: DefId) -> stable_mir::ty::AliasDef {
    with_tables(|t| t.alias_def(did))
}

pub fn param_def(did: DefId) -> stable_mir::ty::ParamDef {
    with_tables(|t| t.param_def(did))
}

pub fn br_named_def(did: DefId) -> stable_mir::ty::BrNamedDef {
    with_tables(|t| t.br_named_def(did))
}

impl<'tcx> Tables<'tcx> {
    pub fn item_def_id(&self, item: &stable_mir::CrateItem) -> DefId {
        self.def_ids[item.0]
    }

    pub fn crate_item(&mut self, did: DefId) -> stable_mir::CrateItem {
        stable_mir::CrateItem(self.create_def_id(did))
    }

    pub fn adt_def(&mut self, did: DefId) -> stable_mir::ty::AdtDef {
        stable_mir::ty::AdtDef(self.create_def_id(did))
    }

    pub fn foreign_def(&mut self, did: DefId) -> stable_mir::ty::ForeignDef {
        stable_mir::ty::ForeignDef(self.create_def_id(did))
    }

    pub fn fn_def(&mut self, did: DefId) -> stable_mir::ty::FnDef {
        stable_mir::ty::FnDef(self.create_def_id(did))
    }

    pub fn closure_def(&mut self, did: DefId) -> stable_mir::ty::ClosureDef {
        stable_mir::ty::ClosureDef(self.create_def_id(did))
    }

    pub fn generator_def(&mut self, did: DefId) -> stable_mir::ty::GeneratorDef {
        stable_mir::ty::GeneratorDef(self.create_def_id(did))
    }

    pub fn alias_def(&mut self, did: DefId) -> stable_mir::ty::AliasDef {
        stable_mir::ty::AliasDef(self.create_def_id(did))
    }

    pub fn param_def(&mut self, did: DefId) -> stable_mir::ty::ParamDef {
        stable_mir::ty::ParamDef(self.create_def_id(did))
    }

    pub fn br_named_def(&mut self, did: DefId) -> stable_mir::ty::BrNamedDef {
        stable_mir::ty::BrNamedDef(self.create_def_id(did))
    }

    fn create_def_id(&mut self, did: DefId) -> stable_mir::DefId {
        // FIXME: this becomes inefficient when we have too many ids
        for (i, &d) in self.def_ids.iter().enumerate() {
            if d == did {
                return i;
            }
        }
        let id = self.def_ids.len();
        self.def_ids.push(did);
        id
    }
}

pub fn crate_num(item: &stable_mir::Crate) -> CrateNum {
    item.id.into()
}

pub fn run(tcx: TyCtxt<'_>, f: impl FnOnce()) {
    crate::stable_mir::run(Tables { tcx, def_ids: vec![], types: vec![] }, f);
}

/// A type that provides internal information but that can still be used for debug purpose.
pub type Opaque = impl Debug + ToString + Clone;

pub(crate) fn opaque<T: Debug>(value: &T) -> Opaque {
    format!("{value:?}")
}
