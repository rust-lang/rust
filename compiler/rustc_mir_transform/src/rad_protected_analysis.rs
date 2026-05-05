//! This pass performs an analysis to determine reference, raw pointer, and unsafe function call accesses that are protected by `#[rad_protected]`

use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_hir::find_attr;
use rustc_middle::mir::{
    Body, Local, LocalKind, Operand, Place, RETURN_PLACE, Rvalue, StatementKind, TerminatorKind,
};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, TyCtxt};

pub(super) struct RadProtectedAnalysis;

impl<'tcx> crate::MirPass<'tcx> for RadProtectedAnalysis {
    fn is_enabled(&self, _sess: &rustc_session::Session) -> bool {
        true
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();

        // print_calls_to_protected_functions(tcx, body);

        if !find_attr!(tcx, def_id, RadProtected(_)) {
            return;
        }

        let sources = build_pointer_sources(body);

        with_no_trimmed_paths!({
            let fn_name = tcx.def_path_str(def_id);
            let loc = format_span(tcx, tcx.def_span(def_id));

            eprintln!("\n=== Protected Function Analysis ===");
            eprintln!("name: {}", fn_name);
            eprintln!("span: {}", loc);

            eprintln!("\nReturn:");
            print_return(body, RETURN_PLACE, &sources);

            eprintln!("\nInputs:");
            for local in body.args_iter() {
                print_input(body, local, &sources);
            }

            eprintln!("\nUser Locals:");
            for local in body.vars_and_temps_iter() {
                if body.local_decls[local].is_user_variable() {
                    let ty = body.local_decls[local].ty;
                    eprintln!("  {:?}: {}", local, ty);
                }
            }

            eprintln!("\nPointer/Reference aliases:");
            print_pointer_aliases(body, &sources);

            eprintln!("\nCalls:");
            print_calls(tcx, body);
        });
    }

    fn is_required(&self) -> bool {
        true
    }
}

enum PointerSource<'tcx> {
    InputRef(Local),
    InputRaw(Local),
    Place { root: Place<'tcx>, via: &'static str },
    AliasOf { root: Local, via: &'static str },
    Unknown,
}

fn _print_calls_to_protected_functions<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
    for (_bb_idx, bb_data) in body.basic_blocks.iter_enumerated() {
        let Some(terminator) = &bb_data.terminator else {
            continue;
        };

        let TerminatorKind::Call { func, args, .. } = &terminator.kind else {
            continue;
        };

        let Some(callee_def_id) = called_def_id(func) else {
            continue;
        };

        if !find_attr!(tcx, callee_def_id, RadProtected(_)) {
            continue;
        }

        with_no_trimmed_paths!({
            let caller_def_id = body.source.def_id();
            let caller_name = tcx.def_path_str(caller_def_id);
            let sources = build_pointer_sources(body);

            eprintln!("\n=== Call To Protected Function ===");
            eprintln!("caller: {}", caller_name);
            eprintln!("callee: {}", tcx.def_path_str(callee_def_id));
            eprintln!("span: {}", format_span(tcx, terminator.source_info.span));

            eprintln!("args:");
            for (idx, arg) in args.iter().enumerate() {
                _print_call_arg_with_source(tcx, body, idx, &arg.node, &sources);
            }
        });
    }
}

fn format_span(tcx: TyCtxt<'_>, span: rustc_span::Span) -> String {
    if span.is_dummy() {
        return "<unknown>".to_string();
    }

    let source_map = tcx.sess.source_map();

    let file = source_map.span_to_filename(span).prefer_local_unconditionally().to_string();

    let line = source_map.lookup_char_pos(span.lo()).line;

    format!("{file}:{line}")
}

fn _print_call_arg_with_source<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    idx: usize,
    operand: &Operand<'tcx>,
    sources: &FxIndexMap<Local, PointerSource<'tcx>>,
) {
    eprintln!("  arg{}: {}", idx, format_operand(operand));

    let (Operand::Copy(place) | Operand::Move(place)) = operand else {
        return;
    };

    let ty = place.ty(body, tcx).ty;

    if matches!(ty.kind(), ty::Ref(..) | ty::RawPtr(..)) {
        eprintln!("\ttype: {}", ty);

        if place.projection.is_empty() {
            match sources.get(&place.local) {
                Some(_) => eprintln!("\tsource: {}", resolve_pointer_source(place.local, sources)),
                None => eprintln!("\tsource: unknown"),
            }
        } else {
            eprintln!("\tsource: {:?}", place);
        }
    }
}

fn build_pointer_sources<'tcx>(body: &Body<'tcx>) -> FxIndexMap<Local, PointerSource<'tcx>> {
    let mut sources = FxIndexMap::default();

    for arg in body.args_iter() {
        let ty = body.local_decls[arg].ty;

        match ty.kind() {
            ty::Ref(..) => {
                sources.insert(arg, PointerSource::InputRef(arg));
            }
            ty::RawPtr(..) => {
                sources.insert(arg, PointerSource::InputRaw(arg));
            }
            _ => {}
        }
    }

    for (_bb_idx, bb_data) in body.basic_blocks.iter_enumerated() {
        for stmt in &bb_data.statements {
            let StatementKind::Assign(box (lhs, rhs)) = &stmt.kind else {
                continue;
            };

            if !lhs.projection.is_empty() {
                continue;
            }

            let lhs_ty = body.local_decls[lhs.local].ty;

            if !matches!(lhs_ty.kind(), ty::Ref(..) | ty::RawPtr(..)) {
                continue;
            }

            match rhs {
                // x = &y
                Rvalue::Ref(_, _, place) => {
                    sources.insert(lhs.local, PointerSource::Place { root: *place, via: "ref" });
                }

                // x = &raw y
                Rvalue::RawPtr(_, place) => {
                    sources.insert(lhs.local, PointerSource::Place { root: *place, via: "&raw" });
                }

                // x = y (copy or move)
                Rvalue::Use(Operand::Copy(place)) | Rvalue::Use(Operand::Move(place)) => {
                    if place.projection.is_empty() {
                        sources.insert(
                            lhs.local,
                            PointerSource::AliasOf { root: place.local, via: "copy/move" },
                        );
                    } else {
                        sources.insert(lhs.local, PointerSource::Unknown);
                    }
                }

                // x = y as *mut T
                Rvalue::Cast(_, operand, _) => match operand {
                    Operand::Copy(place) | Operand::Move(place) => {
                        if place.projection.is_empty() {
                            sources.insert(
                                lhs.local,
                                PointerSource::AliasOf { root: place.local, via: "cast" },
                            );
                        } else {
                            sources.insert(lhs.local, PointerSource::Unknown);
                        }
                    }
                    Operand::Constant(_) | Operand::RuntimeChecks(_) => {
                        sources.insert(lhs.local, PointerSource::Unknown);
                    }
                },

                _ => {
                    sources.insert(lhs.local, PointerSource::Unknown);
                }
            }
        }
    }

    sources
}

fn print_return<'tcx>(
    body: &Body<'tcx>,
    local: Local,
    sources: &FxIndexMap<Local, PointerSource<'tcx>>,
) {
    let ty = body.local_decls[local].ty;

    eprintln!("  {:?}: {}", local, ty);

    if matches!(ty.kind(), ty::Ref(..) | ty::RawPtr(..)) {
        match sources.get(&local) {
            Some(_) => eprintln!("\tsource: {}", resolve_pointer_source(local, sources)),
            None => eprintln!("\tsource: unknown"),
        }
    }
}

fn print_calls<'tcx>(_tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
    for (_bb_idx, bb_data) in body.basic_blocks.iter_enumerated() {
        let Some(terminator) = &bb_data.terminator else {
            continue;
        };

        if let TerminatorKind::Call { func, args, destination, .. } = &terminator.kind {
            eprintln!("func: {}", format_operand(func));

            eprintln!("\targs:");
            for arg in args {
                eprintln!("\t  {}", format_operand(&arg.node));
            }

            eprintln!("\tdest: {:?}", destination);

            if let Some(def_id) = called_def_id(func) {
                let category = if def_id.krate == rustc_hir::def_id::LOCAL_CRATE {
                    "same_crate"
                } else {
                    "external_crate"
                };

                eprintln!("\tcategory: {}", category);
            }
        }
    }
}

fn format_operand<'tcx>(operand: &Operand<'tcx>) -> String {
    match operand {
        Operand::Copy(place) => format!("copy {:?} (by_copy)", place),
        Operand::Move(place) => format!("move {:?} (by_value)", place),
        Operand::Constant(c) => format!("const {:?}", c),
        other => format!("{:?}", other),
    }
}

fn called_def_id<'tcx>(func: &Operand<'tcx>) -> Option<rustc_hir::def_id::DefId> {
    let Operand::Constant(c) = func else {
        return None;
    };

    match c.const_.ty().kind() {
        ty::FnDef(def_id, _) => Some(*def_id),
        _ => None,
    }
}

fn print_input<'tcx>(
    body: &Body<'tcx>,
    local: Local,
    sources: &FxIndexMap<Local, PointerSource<'tcx>>,
) {
    let ty = body.local_decls[local].ty;

    eprintln!("  {:?}: {}", local, ty);

    match ty.kind() {
        ty::Ref(..) | ty::RawPtr(..) => {
            eprintln!("    source: {}", resolve_pointer_source(local, sources));
        }
        _ => {
            eprintln!("    source: by_value");
        }
    }
}

fn print_pointer_aliases<'tcx>(
    body: &Body<'tcx>,
    sources: &FxIndexMap<Local, PointerSource<'tcx>>,
) {
    let mut printed_any = false;

    for local in body.local_decls.indices() {
        let is_formal_arg = matches!(body.local_kind(local), LocalKind::Arg);
        if !body.local_decls[local].is_user_variable() && !is_formal_arg {
            continue;
        }

        let ty = body.local_decls[local].ty;

        if !matches!(ty.kind(), ty::Ref(..) | ty::RawPtr(..)) {
            continue;
        }

        let Some(source) = sources.get(&local) else {
            continue;
        };

        match source {
            PointerSource::Place { via, .. } => {
                printed_any = true;
                eprintln!(
                    "  {:?}: {} -> source {}",
                    local,
                    ty,
                    resolve_pointer_source(local, sources)
                );
                eprintln!("    via: {}", via);
            }

            PointerSource::AliasOf { root, via } => {
                printed_any = true;
                eprintln!(
                    "  {:?}: {} -> source {}",
                    local,
                    ty,
                    resolve_pointer_source(local, sources)
                );
                eprintln!("    via: {} from {:?}", via, root);
            }

            PointerSource::Unknown => {
                printed_any = true;
                eprintln!("  {:?}: {} -> source unknown", local, ty);
                eprintln!("    via: unknown");
            }

            PointerSource::InputRef(_) | PointerSource::InputRaw(_) => {
                printed_any = true;
                eprintln!(
                    "  {:?}: {} -> source {}",
                    local,
                    ty,
                    resolve_pointer_source(local, sources)
                );
                eprintln!("    via: argument");
            }
        }
    }

    if !printed_any {
        eprintln!("  none");
    }
}

fn resolve_pointer_source<'tcx>(
    local: Local,
    sources: &FxIndexMap<Local, PointerSource<'tcx>>,
) -> String {
    let mut current = local;
    let mut seen = FxHashSet::default();

    loop {
        if !seen.insert(current) {
            return format!("cycle_at({:?})", current);
        }

        match sources.get(&current) {
            Some(PointerSource::Place { root, .. }) => {
                return format!("{:?}", root);
            }

            Some(PointerSource::AliasOf { root, .. }) => {
                current = *root;
            }

            Some(PointerSource::InputRef(arg)) => {
                return format!("external_ref_arg({:?})", arg);
            }

            Some(PointerSource::InputRaw(arg)) => {
                return format!("unknown_external_raw_arg({:?})", arg);
            }

            Some(PointerSource::Unknown) | None => {
                return "unknown".to_string();
            }
        }
    }
}
