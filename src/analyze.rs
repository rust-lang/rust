use crate::prelude::*;

use rustc::mir::StatementKind::*;

bitflags::bitflags! {
    pub struct Flags: u8 {
        const NOT_SSA = 0b00000001;
    }
}

pub fn analyze<'a, 'tcx: 'a>(fx: &FunctionCx<'a, 'tcx, impl Backend>) -> HashMap<Local, Flags> {
    let mut flag_map = HashMap::new();

    for local in fx.mir.local_decls.indices() {
        flag_map.insert(local, Flags::empty());
    }

    not_ssa(&mut flag_map, RETURN_PLACE);

    for (local, local_decl) in fx.mir.local_decls.iter_enumerated() {
        if fx.clif_type(local_decl.ty).is_none() {
            not_ssa(&mut flag_map, local);
        }
    }

    for bb in fx.mir.basic_blocks().iter() {
        for stmt in bb.statements.iter() {
            match &stmt.kind {
                Assign(_, rval) => match &**rval {
                    Rvalue::Ref(_, _, place) => analyze_non_ssa_place(&mut flag_map, place),
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

fn analyze_non_ssa_place(flag_map: &mut HashMap<Local, Flags>, place: &Place) {
    match place.base {
        PlaceBase::Local(local) => not_ssa(flag_map, local),
        _ => {}
    }
}

fn not_ssa<L: ::std::borrow::Borrow<Local>>(flag_map: &mut HashMap<Local, Flags>, local: L) {
    *flag_map.get_mut(local.borrow()).unwrap() |= Flags::NOT_SSA;
}
