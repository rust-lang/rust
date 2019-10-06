use crate::prelude::*;

use rustc::mir::StatementKind::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SsaKind {
    NotSsa,
    Ssa,
}

pub fn analyze(fx: &FunctionCx<'_, '_, impl Backend>) -> HashMap<Local, SsaKind> {
    let mut flag_map = HashMap::new();

    for (local, local_decl) in fx.mir.local_decls.iter_enumerated() {
        if fx.clif_type(local_decl.ty).is_some() {
            flag_map.insert(local, SsaKind::Ssa);
        } else {
            flag_map.insert(local, SsaKind::NotSsa);
        }
    }

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

        match &bb.terminator().kind {
            TerminatorKind::Call {
                destination: Some((place, _)),
                ..
            } => analyze_non_ssa_place(&mut flag_map, place),
            _ => {}
        }
    }

    flag_map
}

fn analyze_non_ssa_place(flag_map: &mut HashMap<Local, SsaKind>, place: &Place) {
    match place.base {
        PlaceBase::Local(local) => not_ssa(flag_map, local),
        _ => {}
    }
}

fn not_ssa<L: ::std::borrow::Borrow<Local>>(flag_map: &mut HashMap<Local, SsaKind>, local: L) {
    *flag_map.get_mut(local.borrow()).unwrap() = SsaKind::NotSsa;
}
