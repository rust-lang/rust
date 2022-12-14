//! SSA analysis

use crate::prelude::*;

use rustc_index::vec::IndexVec;
use rustc_middle::mir::StatementKind::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum SsaKind {
    NotSsa,
    Ssa,
}

pub(crate) fn analyze(fx: &FunctionCx<'_, '_, '_>) -> IndexVec<Local, SsaKind> {
    let mut flag_map = fx
        .mir
        .local_decls
        .iter()
        .map(|local_decl| {
            let ty = fx.monomorphize(local_decl.ty);
            if fx.clif_type(ty).is_some() || fx.clif_pair_type(ty).is_some() {
                SsaKind::Ssa
            } else {
                SsaKind::NotSsa
            }
        })
        .collect::<IndexVec<Local, SsaKind>>();

    for bb in fx.mir.basic_blocks.iter() {
        for stmt in bb.statements.iter() {
            match &stmt.kind {
                Assign(place_and_rval) => match &place_and_rval.1 {
                    Rvalue::Ref(_, _, place) | Rvalue::AddressOf(_, place) => {
                        not_ssa(&mut flag_map, place.local)
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }

    flag_map
}

fn not_ssa(flag_map: &mut IndexVec<Local, SsaKind>, local: Local) {
    flag_map[local] = SsaKind::NotSsa;
}
