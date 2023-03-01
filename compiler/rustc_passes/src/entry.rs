use rustc_ast::entry::EntryPointType;
use rustc_errors::error_code;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId, CRATE_DEF_ID, LOCAL_CRATE};
use rustc_hir::{ItemId, Node, CRATE_HIR_ID};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{sigpipe, CrateType, EntryFnType};
use rustc_session::parse::feature_err;
use rustc_span::symbol::sym;
use rustc_span::{Span, Symbol};

use crate::errors::{
    AttrOnlyInFunctions, AttrOnlyOnMain, AttrOnlyOnRootMain, ExternMain, MultipleRustcMain,
    MultipleStartFunctions, NoMainErr, UnixSigpipeValues,
};

struct EntryContext<'tcx> {
    tcx: TyCtxt<'tcx>,

    /// The function that has attribute named `main`.
    attr_main_fn: Option<(LocalDefId, Span)>,

    /// The function that has the attribute 'start' on it.
    start_fn: Option<(LocalDefId, Span)>,

    /// The functions that one might think are `main` but aren't, e.g.
    /// main functions not defined at the top level. For diagnostics.
    non_main_fns: Vec<Span>,
}

fn entry_fn(tcx: TyCtxt<'_>, (): ()) -> Option<(DefId, EntryFnType)> {
    let any_exe = tcx.sess.crate_types().iter().any(|ty| *ty == CrateType::Executable);
    if !any_exe {
        // No need to find a main function.
        return None;
    }

    // If the user wants no main function at all, then stop here.
    if tcx.sess.contains_name(&tcx.hir().attrs(CRATE_HIR_ID), sym::no_main) {
        return None;
    }

    let mut ctxt =
        EntryContext { tcx, attr_main_fn: None, start_fn: None, non_main_fns: Vec::new() };

    for id in tcx.hir().items() {
        find_item(id, &mut ctxt);
    }

    configure_main(tcx, &ctxt)
}

// Beware, this is duplicated in `librustc_builtin_macros/test_harness.rs`
// (with `ast::Item`), so make sure to keep them in sync.
// A small optimization was added so that hir::Item is fetched only when needed.
// An equivalent optimization was not applied to the duplicated code in test_harness.rs.
fn entry_point_type(ctxt: &EntryContext<'_>, id: ItemId, at_root: bool) -> EntryPointType {
    let attrs = ctxt.tcx.hir().attrs(id.hir_id());
    if ctxt.tcx.sess.contains_name(attrs, sym::start) {
        EntryPointType::Start
    } else if ctxt.tcx.sess.contains_name(attrs, sym::rustc_main) {
        EntryPointType::RustcMainAttr
    } else {
        if let Some(name) = ctxt.tcx.opt_item_name(id.owner_id.to_def_id())
            && name == sym::main {
            if at_root {
                // This is a top-level function so can be `main`.
                EntryPointType::MainNamed
            } else {
                EntryPointType::OtherMain
            }
        } else {
            EntryPointType::None
        }
    }
}

fn attr_span_by_symbol(ctxt: &EntryContext<'_>, id: ItemId, sym: Symbol) -> Option<Span> {
    let attrs = ctxt.tcx.hir().attrs(id.hir_id());
    ctxt.tcx.sess.find_by_name(attrs, sym).map(|attr| attr.span)
}

fn find_item(id: ItemId, ctxt: &mut EntryContext<'_>) {
    let at_root = ctxt.tcx.opt_local_parent(id.owner_id.def_id) == Some(CRATE_DEF_ID);

    match entry_point_type(ctxt, id, at_root) {
        EntryPointType::None => {
            if let Some(span) = attr_span_by_symbol(ctxt, id, sym::unix_sigpipe) {
                ctxt.tcx.sess.emit_err(AttrOnlyOnMain { span, attr: sym::unix_sigpipe });
            }
        }
        _ if !matches!(ctxt.tcx.def_kind(id.owner_id), DefKind::Fn) => {
            for attr in [sym::start, sym::rustc_main] {
                if let Some(span) = attr_span_by_symbol(ctxt, id, attr) {
                    ctxt.tcx.sess.emit_err(AttrOnlyInFunctions { span, attr });
                }
            }
        }
        EntryPointType::MainNamed => (),
        EntryPointType::OtherMain => {
            if let Some(span) = attr_span_by_symbol(ctxt, id, sym::unix_sigpipe) {
                ctxt.tcx.sess.emit_err(AttrOnlyOnRootMain { span, attr: sym::unix_sigpipe });
            }
            ctxt.non_main_fns.push(ctxt.tcx.def_span(id.owner_id));
        }
        EntryPointType::RustcMainAttr => {
            if ctxt.attr_main_fn.is_none() {
                ctxt.attr_main_fn = Some((id.owner_id.def_id, ctxt.tcx.def_span(id.owner_id)));
            } else {
                ctxt.tcx.sess.emit_err(MultipleRustcMain {
                    span: ctxt.tcx.def_span(id.owner_id.to_def_id()),
                    first: ctxt.attr_main_fn.unwrap().1,
                    additional: ctxt.tcx.def_span(id.owner_id.to_def_id()),
                });
            }
        }
        EntryPointType::Start => {
            if let Some(span) = attr_span_by_symbol(ctxt, id, sym::unix_sigpipe) {
                ctxt.tcx.sess.emit_err(AttrOnlyOnMain { span, attr: sym::unix_sigpipe });
            }
            if ctxt.start_fn.is_none() {
                ctxt.start_fn = Some((id.owner_id.def_id, ctxt.tcx.def_span(id.owner_id)));
            } else {
                ctxt.tcx.sess.emit_err(MultipleStartFunctions {
                    span: ctxt.tcx.def_span(id.owner_id),
                    labeled: ctxt.tcx.def_span(id.owner_id.to_def_id()),
                    previous: ctxt.start_fn.unwrap().1,
                });
            }
        }
    }
}

fn configure_main(tcx: TyCtxt<'_>, visitor: &EntryContext<'_>) -> Option<(DefId, EntryFnType)> {
    if let Some((def_id, _)) = visitor.start_fn {
        Some((def_id.to_def_id(), EntryFnType::Start))
    } else if let Some((local_def_id, _)) = visitor.attr_main_fn {
        let def_id = local_def_id.to_def_id();
        Some((def_id, EntryFnType::Main { sigpipe: sigpipe(tcx, def_id) }))
    } else {
        if let Some(main_def) = tcx.resolutions(()).main_def && let Some(def_id) = main_def.opt_fn_def_id() {
            // non-local main imports are handled below
            if let Some(def_id) = def_id.as_local() && matches!(tcx.hir().find_by_def_id(def_id), Some(Node::ForeignItem(_))) {
                tcx.sess.emit_err(ExternMain { span: tcx.def_span(def_id) });
                return None;
            }

            if main_def.is_import && !tcx.features().imported_main {
                let span = main_def.span;
                feature_err(
                    &tcx.sess.parse_sess,
                    sym::imported_main,
                    span,
                    "using an imported function as entry point `main` is experimental",
                )
                .emit();
            }
            return Some((def_id, EntryFnType::Main { sigpipe: sigpipe(tcx, def_id) }));
        }
        no_main_err(tcx, visitor);
        None
    }
}

fn sigpipe(tcx: TyCtxt<'_>, def_id: DefId) -> u8 {
    if let Some(attr) = tcx.get_attr(def_id, sym::unix_sigpipe) {
        match (attr.value_str(), attr.meta_item_list()) {
            (Some(sym::inherit), None) => sigpipe::INHERIT,
            (Some(sym::sig_ign), None) => sigpipe::SIG_IGN,
            (Some(sym::sig_dfl), None) => sigpipe::SIG_DFL,
            (_, Some(_)) => {
                // Keep going so that `fn emit_malformed_attribute()` can print
                // an excellent error message
                sigpipe::DEFAULT
            }
            _ => {
                tcx.sess.emit_err(UnixSigpipeValues { span: attr.span });
                sigpipe::DEFAULT
            }
        }
    } else {
        sigpipe::DEFAULT
    }
}

fn no_main_err(tcx: TyCtxt<'_>, visitor: &EntryContext<'_>) {
    let sp = tcx.def_span(CRATE_DEF_ID);
    if *tcx.sess.parse_sess.reached_eof.borrow() {
        // There's an unclosed brace that made the parser reach `Eof`, we shouldn't complain about
        // the missing `fn main()` then as it might have been hidden inside an unclosed block.
        tcx.sess.delay_span_bug(sp, "`main` not found, but expected unclosed brace error");
        return;
    }

    // There is no main function.
    let mut has_filename = true;
    let filename = tcx.sess.local_crate_source_file().unwrap_or_else(|| {
        has_filename = false;
        Default::default()
    });
    let main_def_opt = tcx.resolutions(()).main_def;
    let diagnostic_id = error_code!(E0601);
    let add_teach_note = tcx.sess.teach(&diagnostic_id);
    // The file may be empty, which leads to the diagnostic machinery not emitting this
    // note. This is a relatively simple way to detect that case and emit a span-less
    // note instead.
    let file_empty = !tcx.sess.source_map().lookup_line(sp.hi()).is_ok();

    tcx.sess.emit_err(NoMainErr {
        sp,
        crate_name: tcx.crate_name(LOCAL_CRATE),
        has_filename,
        filename,
        file_empty,
        non_main_fns: visitor.non_main_fns.clone(),
        main_def_opt,
        add_teach_note,
    });
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers { entry_fn, ..*providers };
}
