//! Flow analysis and reporting for `#[rad_protected]` functions.

use std::sync::Mutex;

use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap};
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::find_attr;
use rustc_middle::mir::{
    Body, Local, Operand, RETURN_PLACE, Rvalue, StatementKind, TerminatorKind,
};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, Ty, TyCtxt};

// Pass 1: local summary (runs on every MIR body)
pub(super) struct RadProtectedLocalSummary;

impl<'tcx> crate::MirPass<'tcx> for RadProtectedLocalSummary {
    fn is_enabled(&self, _sess: &rustc_session::Session) -> bool {
        true
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();
        if !def_id.is_local() {
            return;
        }
        let body: &'tcx Body<'tcx> = unsafe { std::mem::transmute(body) };
        let summary = build_local_summary(tcx, body);
        SummaryCache::insert(tcx, def_id, summary);
    }

    fn is_required(&self) -> bool {
        false
    }
}

// Pass 2: protected propagation + report (runs on #[rad_protected] roots only)
pub(super) struct RadProtectedReport;

impl<'tcx> crate::MirPass<'tcx> for RadProtectedReport {
    fn is_enabled(&self, _sess: &rustc_session::Session) -> bool {
        true
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let root = body.source.def_id();
        if !find_attr!(tcx, root, RadProtected(_)) {
            return;
        }

        let body: &'tcx Body<'tcx> = unsafe { std::mem::transmute(body) };
        let root_summary = build_local_summary(tcx, body);
        SummaryCache::insert(tcx, root, root_summary.clone());

        with_no_trimmed_paths!({
            let report = build_report(tcx, root, root_summary);
            print_report(tcx, &report);
        });
    }

    fn is_required(&self) -> bool {
        false
    }
}
// Summary cache (populated by pass 1, read by pass 2)
struct SummaryCache;

static SUMMARY_CACHE: Mutex<Option<(usize, FxHashMap<DefId, LocalMirSummary<'static>>)>> =
    Mutex::new(None);

impl SummaryCache {
    fn tcx_token(tcx: TyCtxt<'_>) -> usize {
        std::ptr::from_ref(&tcx).addr()
    }

    fn ensure(tcx: TyCtxt<'_>) {
        let token = Self::tcx_token(tcx);
        let mut cache = SUMMARY_CACHE.lock().unwrap();
        if cache.as_ref().is_none_or(|(t, _)| *t != token) {
            *cache = Some((token, FxHashMap::default()));
        }
    }

    fn insert<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, summary: LocalMirSummary<'tcx>) {
        Self::ensure(tcx);
        let summary: LocalMirSummary<'static> = unsafe { std::mem::transmute(summary) };
        SUMMARY_CACHE.lock().unwrap().as_mut().unwrap().1.insert(def_id, summary);
    }

    fn get<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> Option<LocalMirSummary<'tcx>> {
        Self::ensure(tcx);
        let cache = SUMMARY_CACHE.lock().unwrap();
        let summary = cache.as_ref()?.1.get(&def_id)?;
        Some(unsafe { std::mem::transmute(summary.clone()) })
    }

    fn get_or_build<'tcx>(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        root_in_flight: DefId,
        warnings: &mut Vec<String>,
    ) -> Option<LocalMirSummary<'tcx>> {
        if let Some(summary) = Self::get(tcx, def_id) {
            return Some(summary);
        }
        if def_id == root_in_flight {
            return None;
        }
        let body = try_get_mir(tcx, def_id, warnings)?;
        let summary = build_local_summary(tcx, body);
        Self::insert(tcx, def_id, summary.clone());
        Some(summary)
    }
}

// Local MIR summary (pass 1)
#[derive(Clone, Debug)]
enum LocalProvenance {
    FormalArg { index: usize },
    CopyMove(Local),
    CallReturn { callsite: usize },
    Constant,
    Ref { from: Local },
    RawPtr { from: Local },
    Derived { from: Local },
    DerivedFrom(Vec<Local>),
    Unknown,
}

#[derive(Clone, Debug)]
enum CalleeTarget {
    Direct(DefId),
    Indirect,
    ExternalDirect(DefId),
}

#[derive(Clone, Debug)]
struct CallSiteSummary {
    callee: CalleeTarget,
    args: Vec<CallArg>,
    _destination: Local,
    _span_label: String,
}

#[derive(Clone, Debug)]
struct CallArg {
    local: Option<Local>,
    constant: bool,
}

#[derive(Clone, Debug)]
enum ReturnSource {
    Arg(Local),
    CallReturn { callsite: usize },
    CopyMove(Local),
    Unknown,
}

#[derive(Clone)]
struct LocalMirSummary<'tcx> {
    formal_args: Vec<(Local, Ty<'tcx>)>,
    return_ty: Ty<'tcx>,
    callsites: Vec<CallSiteSummary>,
    local_provenance: FxIndexMap<Local, LocalProvenance>,
    return_source: ReturnSource,
}

fn build_local_summary<'tcx>(tcx: TyCtxt<'tcx>, body: &'tcx Body<'tcx>) -> LocalMirSummary<'tcx> {
    let mut local_provenance = FxIndexMap::default();

    for (index, arg) in body.args_iter().enumerate() {
        local_provenance.insert(arg, LocalProvenance::FormalArg { index });
    }

    let mut callsites = Vec::new();

    for (_bb, bb_data) in body.basic_blocks.iter_enumerated() {
        for stmt in &bb_data.statements {
            let StatementKind::Assign(box (lhs, rhs)) = &stmt.kind else {
                continue;
            };

            if !lhs.projection.is_empty() {
                continue;
            }

            let prov = match rhs {
                Rvalue::Use(Operand::Copy(place)) | Rvalue::Use(Operand::Move(place)) => {
                    if place.projection.is_empty() {
                        LocalProvenance::CopyMove(place.local)
                    } else {
                        LocalProvenance::Unknown
                    }
                }
                Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place)
                    if place.projection.is_empty() =>
                {
                    if matches!(rhs, Rvalue::Ref(..)) {
                        LocalProvenance::Ref { from: place.local }
                    } else {
                        LocalProvenance::RawPtr { from: place.local }
                    }
                }
                Rvalue::Ref(_, _, _) | Rvalue::RawPtr(_, _) => LocalProvenance::Unknown,
                Rvalue::Cast(_, operand, _) => operand_provenance(operand),
                Rvalue::Aggregate(_, ops) => {
                    if ops.iter().all(|op| matches!(op, Operand::Constant(_))) {
                        LocalProvenance::Constant
                    } else {
                        derived_from_operands(ops.iter())
                    }
                }
                Rvalue::UnaryOp(_, op) => operand_provenance(op),
                Rvalue::BinaryOp(_, box (op1, op2)) => {
                    derived_from_operands([op1, op2].into_iter())
                }
                Rvalue::Discriminant(place) if place.projection.is_empty() => {
                    LocalProvenance::Derived { from: place.local }
                }
                Rvalue::Repeat(..) | Rvalue::ThreadLocalRef(_) | Rvalue::WrapUnsafeBinder(..)
                | Rvalue::CopyForDeref(_) => LocalProvenance::Unknown,
                _ => LocalProvenance::Unknown,
            };

            local_provenance.insert(lhs.local, prov);
        }

        let Some(terminator) = &bb_data.terminator else {
            continue;
        };

        if let TerminatorKind::Call { func, args, destination, .. } = &terminator.kind {
            let callee = match called_def_id(func) {
                Some(callee) if callee.krate == LOCAL_CRATE && callee.is_local() => {
                    CalleeTarget::Direct(callee)
                }
                Some(callee) => CalleeTarget::ExternalDirect(callee),
                None => CalleeTarget::Indirect,
            };

            let call_args = args
                .iter()
                .map(|arg| match &arg.node {
                    Operand::Constant(_) | Operand::RuntimeChecks(_) => {
                        CallArg { local: None, constant: true }
                    }
                    Operand::Copy(place) | Operand::Move(place) if place.projection.is_empty() => {
                        CallArg { local: Some(place.local), constant: false }
                    }
                    _ => CallArg { local: None, constant: false },
                })
                .collect();

            let callsite_idx = callsites.len();
            callsites.push(CallSiteSummary {
                callee,
                args: call_args,
                _destination: destination.local,
                _span_label: format_span(tcx, terminator.source_info.span),
            });

            if destination.projection.is_empty() {
                local_provenance.insert(
                    destination.local,
                    LocalProvenance::CallReturn { callsite: callsite_idx },
                );
            }
        }
    }

    let return_source = infer_return_source(body, &local_provenance);

    LocalMirSummary {
        formal_args: body
            .args_iter()
            .map(|local| (local, body.local_decls[local].ty))
            .collect(),
        return_ty: body.local_decls[RETURN_PLACE].ty,
        callsites,
        local_provenance,
        return_source,
    }
}

fn operand_local(operand: &Operand<'_>) -> Option<Local> {
    match operand {
        Operand::Copy(place) | Operand::Move(place) if place.projection.is_empty() => {
            Some(place.local)
        }
        Operand::Constant(_) | Operand::RuntimeChecks(_) => None,
        _ => None,
    }
}

fn operand_provenance(operand: &Operand<'_>) -> LocalProvenance {
    match operand {
        Operand::Constant(_) | Operand::RuntimeChecks(_) => LocalProvenance::Constant,
        Operand::Copy(place) | Operand::Move(place) if place.projection.is_empty() => {
            LocalProvenance::Derived { from: place.local }
        }
        _ => LocalProvenance::Unknown,
    }
}

fn derived_from_operands<'a, I>(ops: I) -> LocalProvenance
where
    I: Iterator<Item = &'a Operand<'a>>,
{
    let locals: Vec<Local> = ops.filter_map(operand_local).collect();
    match locals.len() {
        0 => LocalProvenance::Unknown,
        1 => LocalProvenance::Derived { from: locals[0] },
        _ => LocalProvenance::DerivedFrom(locals),
    }
}

fn infer_return_source(body: &Body<'_>, provenance: &FxIndexMap<Local, LocalProvenance>) -> ReturnSource {
    if let Some(prov) = provenance.get(&RETURN_PLACE) {
        return match prov {
            LocalProvenance::CopyMove(local)
            | LocalProvenance::Derived { from: local }
            | LocalProvenance::Ref { from: local }
            | LocalProvenance::RawPtr { from: local } => ReturnSource::CopyMove(*local),
            LocalProvenance::DerivedFrom(_) => ReturnSource::Unknown,
            LocalProvenance::CallReturn { callsite } => ReturnSource::CallReturn { callsite: *callsite },
            LocalProvenance::FormalArg { index } => {
                ReturnSource::Arg(body.args_iter().nth(*index).unwrap_or(RETURN_PLACE))
            }
            _ => ReturnSource::Unknown,
        };
    }
    ReturnSource::Unknown
}

fn try_get_mir<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    _warnings: &mut Vec<String>,
) -> Option<&'tcx Body<'tcx>> {
    let local = def_id.as_local()?;
    if def_id.krate != LOCAL_CRATE {
        return None;
    }
    if !matches!(
        tcx.def_kind(def_id),
        rustc_hir::def::DefKind::Fn
            | rustc_hir::def::DefKind::AssocFn
            | rustc_hir::def::DefKind::Const
            | rustc_hir::def::DefKind::AssocConst
            | rustc_hir::def::DefKind::Closure
            | rustc_hir::def::DefKind::InlineConst
    ) {
        return None;
    }
    Some(tcx.optimized_mir(local))
}

fn called_def_id<'tcx>(func: &Operand<'tcx>) -> Option<DefId> {
    let Operand::Constant(c) = func else {
        return None;
    };
    match c.const_.ty().kind() {
        ty::FnDef(def_id, _) => Some(*def_id),
        _ => None,
    }
}
// Flow sources + interprocedural propagation (pass 2)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum FlowSource {
    ProtectedArg(usize),
    ArgOf { func: DefId, local: Local },
    ReturnOf { func: DefId },
    LocalOf { func: DefId, local: Local },
    DerivedFrom(Vec<FlowSource>),
    Constant,
    Unknown,
}

type ArgKey = (DefId, Local);

struct InterprocGraph {
    reached: Vec<DefId>,
    children: FxHashMap<DefId, Vec<DefId>>,
    ip_sources: FxIndexMap<ArgKey, FlowSource>,
    warnings: Vec<String>,
}

fn build_interproc_graph<'tcx>(
    tcx: TyCtxt<'tcx>,
    root: DefId,
    root_summary: &LocalMirSummary<'tcx>,
) -> InterprocGraph {
    let mut reached = Vec::new();
    let mut children: FxHashMap<DefId, Vec<DefId>> = FxHashMap::default();
    let mut ip_sources = FxIndexMap::default();
    let mut warnings = Vec::new();
    let mut visited = FxHashSet::default();
    let mut externals = Vec::new();
    let mut external_seen = FxHashSet::default();
    let mut worklist = vec![root];

    for (index, (local, _)) in root_summary.formal_args.iter().enumerate() {
        ip_sources.insert((root, *local), FlowSource::ProtectedArg(index));
    }

    while let Some(caller) = worklist.pop() {
        if !visited.insert(caller) {
            continue;
        }
        reached.push(caller);

        let caller_summary = if caller == root {
            root_summary.clone()
        } else if let Some(summary) = SummaryCache::get_or_build(tcx, caller, root, &mut warnings) {
            summary
        } else {
            continue;
        };

        for callsite in &caller_summary.callsites {
            match &callsite.callee {
                CalleeTarget::Direct(callee) => {
                    children.entry(caller).or_default().push(*callee);
                    if !visited.contains(callee) {
                        worklist.push(*callee);
                    }

                    let Some(callee_summary) =
                        SummaryCache::get_or_build(tcx, *callee, root, &mut warnings)
                    else {
                        continue;
                    };

                    for (arg_idx, (callee_local, _)) in callee_summary.formal_args.iter().enumerate() {
                        let callee_key = (*callee, *callee_local);
                        let source = match callsite.args.get(arg_idx) {
                            Some(CallArg { constant: true, .. }) => FlowSource::Constant,
                            Some(CallArg { local: Some(actual), constant: false }) => {
                                resolve_actual_flow(
                                    tcx,
                                    caller,
                                    *actual,
                                    &caller_summary,
                                    &ip_sources,
                                )
                            }
                            _ => FlowSource::Unknown,
                        };

                        ip_sources.insert(callee_key, source.clone());
                    }
                }
                CalleeTarget::ExternalDirect(callee) => {
                    children.entry(caller).or_default().push(*callee);
                    if external_seen.insert(*callee) {
                        externals.push(*callee);
                    }
                }
                CalleeTarget::Indirect => {}
            }
        }
    }

    reached.extend(externals);

    InterprocGraph { reached, children, ip_sources, warnings }
}

fn is_local_body(def_id: DefId) -> bool {
    def_id.is_local() && def_id.krate == LOCAL_CRATE
}

fn sorted_unique_children(
    children: &FxHashMap<DefId, Vec<DefId>>,
    node: DefId,
) -> Vec<DefId> {
    let Some(kids) = children.get(&node) else {
        return Vec::new();
    };
    let mut unique = Vec::new();
    let mut seen = FxHashSet::default();
    for kid in kids {
        if seen.insert(*kid) {
            unique.push(*kid);
        }
    }
    unique.sort_by_key(|id| (id.krate, id.index));
    unique
}

#[derive(Clone)]
struct InputEntry<'tcx> {
    ty: Ty<'tcx>,
    source: FlowSource,
    absolute_source: FlowSource,
}

fn input_entries_for_fn<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    inputs: &FxIndexMap<ArgKey, InputEntry<'tcx>>,
) -> Vec<(Local, InputEntry<'tcx>)> {
    if is_local_body(def_id) {
        let mut v: Vec<_> = inputs
            .iter()
            .filter(|((f, _), _)| *f == def_id)
            .map(|((_, local), entry)| (*local, entry.clone()))
            .collect();
        v.sort_by_key(|(local, _)| local.index());
        v
    } else {
        let sig = tcx.fn_sig(def_id).instantiate_identity().skip_binder();
        sig.inputs()
            .iter()
            .enumerate()
            .map(|(i, ty)| {
                let local = Local::from_usize(i + 1);
                (
                    local,
                    InputEntry {
                        ty: *ty,
                        source: FlowSource::Unknown,
                        absolute_source: FlowSource::Unknown,
                    },
                )
            })
            .collect()
    }
}

fn print_inputs_in_call_tree<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    report: &ProtectedReport<'tcx>,
    visited: &mut FxHashSet<DefId>,
    needs_gap: &mut bool,
) {
    if is_local_body(def_id) && !visited.insert(def_id) {
        return;
    }

    let entries = input_entries_for_fn(tcx, def_id, &report.inputs);
    if !entries.is_empty() {
        if *needs_gap {
            eprintln!();
        }
        *needs_gap = true;

        for (local, entry) in entries {
            let label = format_fn_local(tcx, def_id, local);
            eprintln!("  {label}: {}", entry.ty);
            if is_pointer_like(entry.ty) {
                eprintln!("    source: {}", format_flow_source(tcx, &entry.source));
                eprintln!(
                    "    absolute_source: {}",
                    format_flow_source(tcx, &entry.absolute_source)
                );
            }
        }
    }

    for kid in sorted_unique_children(&report.children, def_id) {
        print_inputs_in_call_tree(tcx, kid, report, visited, needs_gap);
    }
}

fn resolve_actual_flow(
    tcx: TyCtxt<'_>,
    caller: DefId,
    local: Local,
    summary: &LocalMirSummary<'_>,
    ip_sources: &FxIndexMap<ArgKey, FlowSource>,
) -> FlowSource {
    let mut seen_locals = FxHashSet::default();
    resolve_local_at_callsite(tcx, caller, local, summary, ip_sources, &mut seen_locals)
}

fn resolve_local_at_callsite(
    tcx: TyCtxt<'_>,
    func: DefId,
    local: Local,
    summary: &LocalMirSummary<'_>,
    ip_sources: &FxIndexMap<ArgKey, FlowSource>,
    seen_locals: &mut FxHashSet<Local>,
) -> FlowSource {
    if !seen_locals.insert(local) {
        return FlowSource::Unknown;
    }

    if let Some(flow) = ip_sources.get(&(func, local)) {
        return flow.clone();
    }

    match summary.local_provenance.get(&local) {
        Some(LocalProvenance::FormalArg { .. }) => ip_sources
            .get(&(func, local))
            .cloned()
            .unwrap_or(FlowSource::ArgOf { func, local }),
        Some(LocalProvenance::Derived { from }) => FlowSource::DerivedFrom(vec![
            FlowSource::LocalOf { func, local },
            provenance_to_flow(tcx, func, *from, summary, ip_sources, seen_locals),
        ]),
        Some(LocalProvenance::DerivedFrom(locals)) => {
            let mut deps = vec![FlowSource::LocalOf { func, local }];
            deps.extend(
                locals
                    .iter()
                    .map(|l| provenance_to_flow(tcx, func, *l, summary, ip_sources, seen_locals)),
            );
            FlowSource::DerivedFrom(deps)
        }
        Some(LocalProvenance::CopyMove(from)) => {
            provenance_to_flow(tcx, func, *from, summary, ip_sources, seen_locals)
        }
        Some(LocalProvenance::CallReturn { callsite }) => {
            let callsite = &summary.callsites[*callsite];
            match &callsite.callee {
                CalleeTarget::Direct(callee) => FlowSource::ReturnOf { func: *callee },
                CalleeTarget::ExternalDirect(callee) => FlowSource::ReturnOf { func: *callee },
                CalleeTarget::Indirect => FlowSource::Unknown,
            }
        }
        Some(LocalProvenance::Constant) => FlowSource::Constant,
        Some(LocalProvenance::Ref { from } | LocalProvenance::RawPtr { from }) => {
            FlowSource::DerivedFrom(vec![
                FlowSource::LocalOf { func, local },
                provenance_to_flow(tcx, func, *from, summary, ip_sources, seen_locals),
            ])
        }
        Some(LocalProvenance::Unknown) | None => {
            if summary.formal_args.iter().any(|(l, _)| *l == local) {
                ip_sources
                    .get(&(func, local))
                    .cloned()
                    .unwrap_or(FlowSource::ArgOf { func, local })
            } else {
                FlowSource::LocalOf { func, local }
            }
        }
    }
}

fn provenance_to_flow(
    tcx: TyCtxt<'_>,
    func: DefId,
    local: Local,
    summary: &LocalMirSummary<'_>,
    ip_sources: &FxIndexMap<ArgKey, FlowSource>,
    seen_locals: &mut FxHashSet<Local>,
) -> FlowSource {
    if !seen_locals.insert(local) {
        return FlowSource::Unknown;
    }

    if let Some(flow) = ip_sources.get(&(func, local)) {
        return flow.clone();
    }

    match summary.local_provenance.get(&local) {
        Some(LocalProvenance::FormalArg { .. }) => ip_sources
            .get(&(func, local))
            .cloned()
            .unwrap_or(FlowSource::ArgOf { func, local }),
        Some(LocalProvenance::CopyMove(from)) => {
            provenance_to_flow(tcx, func, *from, summary, ip_sources, seen_locals)
        }
        Some(LocalProvenance::Derived { from }) | Some(LocalProvenance::Ref { from })
        | Some(LocalProvenance::RawPtr { from }) => FlowSource::DerivedFrom(vec![
            provenance_to_flow(tcx, func, *from, summary, ip_sources, seen_locals),
        ]),
        Some(LocalProvenance::DerivedFrom(locals)) => FlowSource::DerivedFrom(
            locals
                .iter()
                .map(|l| provenance_to_flow(tcx, func, *l, summary, ip_sources, seen_locals))
                .collect(),
        ),
        Some(LocalProvenance::CallReturn { callsite }) => {
            let callsite = &summary.callsites[*callsite];
            match &callsite.callee {
                CalleeTarget::Direct(callee) => FlowSource::ReturnOf { func: *callee },
                CalleeTarget::ExternalDirect(callee) => FlowSource::ReturnOf { func: *callee },
                CalleeTarget::Indirect => FlowSource::Unknown,
            }
        }
        Some(LocalProvenance::Constant) => FlowSource::Constant,
        Some(LocalProvenance::Unknown) | None => {
            if summary.formal_args.iter().any(|(l, _)| *l == local) {
                ip_sources
                    .get(&(func, local))
                    .cloned()
                    .unwrap_or(FlowSource::ArgOf { func, local })
            } else {
                FlowSource::LocalOf { func, local }
            }
        }
    }
}

struct OutputEntry<'tcx> {
    ty: Ty<'tcx>,
    source: FlowSource,
}

#[allow(dead_code)]
struct DuplicationPlan {
    duplicate: Vec<String>,
    source_track: Vec<String>,
    compare: Vec<String>,
}

struct ProtectedReport<'tcx> {
    root: DefId,
    reached: Vec<DefId>,
    children: FxHashMap<DefId, Vec<DefId>>,
    inputs: FxIndexMap<ArgKey, InputEntry<'tcx>>,
    outputs: FxIndexMap<ArgKey, OutputEntry<'tcx>>,
    #[allow(dead_code)]
    duplication: DuplicationPlan,
    warnings: Vec<String>,
}

fn build_report<'tcx>(
    tcx: TyCtxt<'tcx>,
    root: DefId,
    root_summary: LocalMirSummary<'tcx>,
) -> ProtectedReport<'tcx> {
    let mut graph = build_interproc_graph(tcx, root, &root_summary);
    let mut inputs = FxIndexMap::default();

    for func in graph.reached.clone() {
        if !is_local_body(func) {
            continue;
        }
        let summary = if func == root {
            root_summary.clone()
        } else if let Some(summary) = SummaryCache::get_or_build(tcx, func, root, &mut graph.warnings) {
            summary
        } else {
            continue;
        };

        for (local, ty) in &summary.formal_args {
            let key = (func, *local);
            let source = graph
                .ip_sources
                .get(&key)
                .cloned()
                .unwrap_or(FlowSource::Unknown);
            let absolute_source =
                resolve_absolute_source(root, source.clone(), &graph.ip_sources);
            inputs.insert(key, InputEntry { ty: *ty, source, absolute_source });
        }
    }

    let mut outputs = FxIndexMap::default();

    for func in graph.reached.clone() {
        let summary = if func == root {
            root_summary.clone()
        } else if let Some(summary) = SummaryCache::get_or_build(tcx, func, root, &mut graph.warnings) {
            summary
        } else {
            continue;
        };
        let output_key = (func, RETURN_PLACE);
        let source = resolve_output_source(tcx, func, &summary, &graph);
        outputs.insert(output_key, OutputEntry { ty: summary.return_ty, source });
    }

    // let duplication = build_duplication_plan(tcx, root, &inputs, &outputs);
    let duplication =
        DuplicationPlan { duplicate: Vec::new(), source_track: Vec::new(), compare: Vec::new() };

    ProtectedReport {
        root,
        reached: graph.reached,
        children: graph.children,
        inputs,
        outputs,
        duplication,
        warnings: graph.warnings,
    }
}

fn resolve_absolute_source(
    root: DefId,
    source: FlowSource,
    ip_sources: &FxIndexMap<ArgKey, FlowSource>,
) -> FlowSource {
    let mut seen = FxHashSet::default();
    resolve_absolute_source_inner(root, source, ip_sources, &mut seen)
}

fn resolve_absolute_source_inner(
    root: DefId,
    source: FlowSource,
    ip_sources: &FxIndexMap<ArgKey, FlowSource>,
    seen: &mut FxHashSet<ArgKey>,
) -> FlowSource {
    match source {
        FlowSource::ArgOf { func, local } => {
            if func == root {
                return FlowSource::ArgOf { func, local };
            }
            let key = (func, local);
            if !seen.insert(key) {
                return FlowSource::ArgOf { func, local };
            }
            let next = ip_sources.get(&key).cloned().unwrap_or(FlowSource::Unknown);
            resolve_absolute_source_inner(root, next, ip_sources, seen)
        }
        FlowSource::DerivedFrom(sources) => FlowSource::DerivedFrom(
            sources
                .iter()
                .map(|s| resolve_absolute_source_inner(root, s.clone(), ip_sources, seen))
                .collect(),
        ),
        FlowSource::LocalOf { func, local } => {
            let key = (func, local);
            if !seen.insert(key) {
                return FlowSource::LocalOf { func, local };
            }
            let next = ip_sources
                .get(&key)
                .cloned()
                .unwrap_or(FlowSource::LocalOf { func, local });
            resolve_absolute_source_inner(root, next, ip_sources, seen)
        }
        other => other,
    }
}

fn resolve_output_source<'tcx>(
    tcx: TyCtxt<'tcx>,
    func: DefId,
    summary: &LocalMirSummary<'tcx>,
    graph: &InterprocGraph,
) -> FlowSource {
    match &summary.return_source {
        ReturnSource::CallReturn { callsite } => {
            let callsite = &summary.callsites[*callsite];
            match callsite.callee {
                CalleeTarget::Direct(callee) | CalleeTarget::ExternalDirect(callee) => {
                    FlowSource::ReturnOf { func: callee }
                }
                CalleeTarget::Indirect => FlowSource::Unknown,
            }
        }
        ReturnSource::CopyMove(local) | ReturnSource::Arg(local) => {
            resolve_actual_flow(tcx, func, *local, summary, &graph.ip_sources)
        }
        ReturnSource::Unknown => FlowSource::Unknown,
    }
}

#[allow(dead_code)]
fn build_duplication_plan<'tcx>(
    tcx: TyCtxt<'_>,
    root: DefId,
    inputs: &FxIndexMap<ArgKey, InputEntry<'tcx>>,
    outputs: &FxIndexMap<ArgKey, OutputEntry<'tcx>>,
) -> DuplicationPlan {
    let mut duplicate = Vec::new();
    let mut source_track = Vec::new();
    let mut compare = Vec::new();

    for ((func, local), entry) in inputs {
        let label = format_fn_local(tcx, *func, *local);
        if is_pointer_like(entry.ty) {
            source_track.push(label);
        } else if entry.ty.is_trivially_pure_clone_copy() {
            duplicate.push(label);
        }
    }

    if outputs.contains_key(&(root, RETURN_PLACE)) {
        compare.push(format_fn_local(tcx, root, RETURN_PLACE));
    }

    DuplicationPlan { duplicate, source_track, compare }
}

fn is_pointer_like(ty: Ty<'_>) -> bool {
    matches!(ty.kind(), ty::Ref(..) | ty::RawPtr(..))
}

fn print_report<'tcx>(tcx: TyCtxt<'tcx>, report: &ProtectedReport<'tcx>) {
    eprintln!("\n=== Protected Function Analysis ===");
    eprintln!("Protected root: {}", tcx.def_path_str(report.root));

    eprintln!("\nReached functions:");
    for func in &report.reached {
        let label = tcx.def_path_str(*func);
        if is_local_body(*func) {
            eprintln!("  - {label}");
        } else {
            eprintln!("  - {label} (external)");
        }
    }

    eprintln!("\nCall tree:");
    let mut print_visited = FxHashSet::default();
    print_call_tree(tcx, report.root, &report.children, 0, &mut print_visited);

    eprintln!("\nInputs:");
    let mut inputs_visited = FxHashSet::default();
    let mut inputs_needs_gap = false;
    print_inputs_in_call_tree(tcx, report.root, report, &mut inputs_visited, &mut inputs_needs_gap);

    eprintln!("\nOutputs:");
    for func in &report.reached {
        let key = (*func, RETURN_PLACE);
        let Some(entry) = report.outputs.get(&key) else {
            continue;
        };
        eprintln!("  {}: {}", format_fn_local(tcx, *func, RETURN_PLACE), entry.ty);
        eprintln!("    source: {}", format_flow_source(tcx, &entry.source));
    }

    /*
    eprintln!("\nDuplication plan:");
    eprintln!("  duplicate:");
    if report.duplication.duplicate.is_empty() {
        eprintln!("    none");
    } else {
        for item in &report.duplication.duplicate {
            eprintln!("    - {}", item);
        }
    }
    eprintln!("  source-track:");
    if report.duplication.source_track.is_empty() {
        eprintln!("    none");
    } else {
        for item in &report.duplication.source_track {
            eprintln!("    - {}", item);
        }
    }
    eprintln!("  compare:");
    if report.duplication.compare.is_empty() {
        eprintln!("    none");
    } else {
        for item in &report.duplication.compare {
            eprintln!("    - {}", item);
        }
    }
    */

    eprintln!("\nWarnings:");
    if report.warnings.is_empty() {
        eprintln!("  none");
    } else {
        for warning in &report.warnings {
            eprintln!("  - {}", warning);
        }
    }
}

fn print_call_tree(
    tcx: TyCtxt<'_>,
    node: DefId,
    children: &FxHashMap<DefId, Vec<DefId>>,
    depth: usize,
    visited: &mut FxHashSet<DefId>,
) {
    let indent = "  ".repeat(depth);
    if depth == 0 {
        eprintln!("  {}", tcx.def_path_str(node));
    }

    if is_local_body(node) {
        if !visited.insert(node) {
            eprintln!("{indent}  └── <cycle: {}>", tcx.def_path_str(node));
            return;
        }
    }

    let unique = sorted_unique_children(children, node);
    if unique.is_empty() {
        return;
    }

    for (idx, kid) in unique.iter().enumerate() {
        let branch = if idx + 1 == unique.len() { "└──" } else { "├──" };
        let name = tcx.def_path_str(*kid);
        if is_local_body(*kid) {
            eprintln!("{indent}  {branch} {name}");
            print_call_tree(tcx, *kid, children, depth + 2, visited);
        } else {
            eprintln!("{indent}  {branch} {name} (external)");
        }
    }
}

fn format_fn_local(tcx: TyCtxt<'_>, def_id: DefId, local: Local) -> String {
    format!("{}._{}", tcx.def_path_str(def_id), local.index())
}

fn format_flow_source(tcx: TyCtxt<'_>, source: &FlowSource) -> String {
    match source {
        FlowSource::ProtectedArg(index) => format!("protected_arg({index})"),
        FlowSource::ArgOf { func, local } => format_fn_local(tcx, *func, *local),
        FlowSource::ReturnOf { func } => format!("return_of({})", tcx.def_path_str(*func)),
        FlowSource::LocalOf { func, local } => format_fn_local(tcx, *func, *local),
        FlowSource::DerivedFrom(sources) => {
            let parts: Vec<_> = sources.iter().map(|s| format_flow_source(tcx, s)).collect();
            format!("derived_from({})", parts.join(", "))
        }
        FlowSource::Constant => "constant".to_string(),
        FlowSource::Unknown => "unknown".to_string(),
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
