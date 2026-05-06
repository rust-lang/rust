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
            print_return(body);

            eprintln!("\nInputs:");
            print_input(body); 

            eprintln!("\nUser Locals:");
            print_local(body);

            eprintln!("\nRaw Pointer Sources:");
            print_raw_pointer_sources(body, &sources);

            eprintln!("\nCalls:");
            print_calls(tcx, body);
        });
    }

    fn is_required(&self) -> bool {
        true
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

fn print_return<'tcx>(body: &Body<'tcx>) {
    let ty = body.local_decls[RETURN_PLACE].ty;

    eprintln!("  {:?}: {}", RETURN_PLACE, ty);
}

fn print_input<'tcx>(body: &Body<'tcx>) {
    for local in body.args_iter() {
        let ty = body.local_decls[local].ty;

        eprintln!("  {:?}: {}", local, ty);

        match ty.kind() {
            ty::Ref(..) => {
                eprintln!("      source: external_ref_arg({:?})", local);
            }
            ty::RawPtr(..) => {
                eprintln!("      source: unknown_external_raw_arg({:?})", local);
            }
            _ => {
                eprintln!("      source: by_value");
            }
        }
    }
}

fn print_local<'tcx>(body: &Body<'tcx>) {
    for local in body.vars_and_temps_iter() {
        let decl = &body.local_decls[local];
        if decl.is_user_variable() && !decl.from_compiler_desugaring() {
            let ty = decl.ty;
            eprintln!("  {:?}: {}", local, ty);
        }
    }
}

fn print_raw_pointer_sources<'tcx>(body: &Body<'tcx>, sources: &FxIndexMap<Local, PointerSource<'tcx>>) {
    let mut printed_any = false;

    for local in body.local_decls.indices() {
        let decl = &body.local_decls[local];
        if !decl.is_user_variable() && !decl.from_compiler_desugaring() && !matches!(body.local_kind(local), LocalKind::Arg)
        {
            continue;
        }

        let ty = decl.ty;

        if !matches!(ty.kind(), ty::RawPtr(..)) {
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

            PointerSource::InputRaw(_) => {
                printed_any = true;
                eprintln!(
                    "  {:?}: {} -> source {}",
                    local,
                    ty,
                    resolve_pointer_source(local, sources)
                );
                eprintln!("    via: argument");
            }

            _ => {}
        }
    }

    if !printed_any {
        eprintln!("  none");
    }
}

fn print_calls<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
    for (_bb_idx, bb_data) in body.basic_blocks.iter_enumerated() {
        let Some(terminator) = &bb_data.terminator else {
            continue;
        };

        if let TerminatorKind::Call { func, args, destination, .. } = &terminator.kind {
            eprintln!("func: {}", format_operand(func, body, tcx));

            eprintln!("\targs:");
            for arg in args {
                eprintln!("\t  {}", format_operand(&arg.node, body, tcx));
            }

            eprintln!("\tdest: {:?}", destination);

            if let Some(def_id) = called_def_id(func) {
                let category = if tcx.is_foreign_item(def_id) {
                    "foreign"
                } else if def_id.krate == rustc_hir::def_id::LOCAL_CRATE {
                    "same_crate"
                } else {
                    "external_crate"
                };

                eprintln!("\tcategory: {}", category);
            }
        }
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

fn format_operand<'tcx>(operand: &Operand<'tcx>, body: &Body<'tcx>, tcx: TyCtxt<'tcx>) -> String {
    if let Some(place) = operand.place() {
        let ty = place.ty(body, tcx).ty;

        if ty.is_trivially_pure_clone_copy() {
            format!("copy {:?}", place)
        } else {
            format!("move {:?}", place)
        }
    } else {
        format!("const {:?}", operand)
    }
}

enum PointerSource<'tcx> {
    InputRef(Local),
    InputRaw(Local),
    Place { root: Place<'tcx>, via: &'static str },
    AliasOf { root: Local, via: &'static str },
    Unknown,
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
                // If the source is another local, keep resolving through it.
                if root.projection.is_empty() {
                    if sources.contains_key(&root.local) {
                        current = root.local;
                        continue;
                    }

                    return format!("{:?}", root);
                }

                if sources.contains_key(&root.local) {
                    current = root.local;
                    continue;
                }

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
