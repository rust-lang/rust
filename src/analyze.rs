use crate::prelude::*;

use rustc::mir::StatementKind::*;
use rustc_index::vec::IndexVec;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SsaKind {
    NotSsa,
    Ssa,
}

pub fn analyze(fx: &FunctionCx<'_, '_, impl Backend>) -> IndexVec<Local, SsaKind> {
    let mut flag_map = fx.mir.local_decls.iter().map(|local_decl| {
        if fx.clif_type(local_decl.ty).is_some() {
            SsaKind::Ssa
        } else {
            SsaKind::NotSsa
        }
    }).collect::<IndexVec<Local, SsaKind>>();

    for bb in fx.mir.basic_blocks().iter() {
        for stmt in bb.statements.iter() {
            match &stmt.kind {
                Assign(place_and_rval) => match &place_and_rval.1 {
                    Rvalue::Ref(_, _, place) => {
                        analyze_non_ssa_place(&mut flag_map, place);
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }

    flag_map
}

fn analyze_non_ssa_place(flag_map: &mut IndexVec<Local, SsaKind>, place: &Place) {
    match place.base {
        PlaceBase::Local(local) => not_ssa(flag_map, local),
        _ => {}
    }
}

fn not_ssa(flag_map: &mut IndexVec<Local, SsaKind>, local: Local) {
    flag_map[local] = SsaKind::NotSsa;
}
