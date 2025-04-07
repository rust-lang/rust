use rustc_ast::attr;
use rustc_ast::entry::EntryPointType;
use rustc_errors::codes::*;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{CRATE_DEF_ID, DefId, LOCAL_CRATE, LocalDefId};
use rustc_hir::{CRATE_HIR_ID, ItemId, Node};
use rustc_middle::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::RemapFileNameExt;
use rustc_session::config::{CrateType, EntryFnType, RemapPathScopeComponents, sigpipe};
use rustc_span::{Span, Symbol, sym};

use crate::errors::{AttrOnlyInFunctions, ExternMain, MultipleRustcMain, NoMainErr};

struct EntryContext<'tcx> {
    tcx: TyCtxt<'tcx>,

    /// The function has the `#[rustc_main]` attribute.
    rustc_main_fn: Option<(LocalDefId, Span)>,

    /// The functions that one might think are `main` but aren't, e.g.
    /// main functions not defined at the top level. For diagnostics.
    non_main_fns: Vec<Span>,
}

fn entry_fn(tcx: TyCtxt<'_>, (): ()) -> Option<(DefId, EntryFnType)> {
    let any_exe = tcx.crate_types().contains(&CrateType::Executable);
    if !any_exe {
        // No need to find a main function.
        return None;
    }

    // If the user wants no main function at all, then stop here.
    if attr::contains_name(tcx.hir_attrs(CRATE_HIR_ID), sym::no_main) {
        return None;
    }

    let mut ctxt = EntryContext { tcx, rustc_main_fn: None, non_main_fns: Vec::new() };

    for id in tcx.hir_free_items() {
        check_and_search_item(id, &mut ctxt);
    }

    configure_main(tcx, &ctxt)
}

fn attr_span_by_symbol(ctxt: &EntryContext<'_>, id: ItemId, sym: Symbol) -> Option<Span> {
    let attrs = ctxt.tcx.hir_attrs(id.hir_id());
    attr::find_by_name(attrs, sym).map(|attr| attr.span())
}

fn check_and_search_item(id: ItemId, ctxt: &mut EntryContext<'_>) {
    if !matches!(ctxt.tcx.def_kind(id.owner_id), DefKind::Fn) {
        for attr in [sym::rustc_main] {
            if let Some(span) = attr_span_by_symbol(ctxt, id, attr) {
                ctxt.tcx.dcx().emit_err(AttrOnlyInFunctions { span, attr });
            }
        }
        return;
    }

    let at_root = ctxt.tcx.opt_local_parent(id.owner_id.def_id) == Some(CRATE_DEF_ID);

    let attrs = ctxt.tcx.hir_attrs(id.hir_id());
    let entry_point_type = rustc_ast::entry::entry_point_type(
        attrs,
        at_root,
        ctxt.tcx.opt_item_name(id.owner_id.to_def_id()),
    );

    match entry_point_type {
        EntryPointType::None => {}
        EntryPointType::MainNamed => {}
        EntryPointType::OtherMain => {
            ctxt.non_main_fns.push(ctxt.tcx.def_span(id.owner_id));
        }
        EntryPointType::RustcMainAttr => {
            if ctxt.rustc_main_fn.is_none() {
                ctxt.rustc_main_fn = Some((id.owner_id.def_id, ctxt.tcx.def_span(id.owner_id)));
            } else {
                ctxt.tcx.dcx().emit_err(MultipleRustcMain {
                    span: ctxt.tcx.def_span(id.owner_id.to_def_id()),
                    first: ctxt.rustc_main_fn.unwrap().1,
                    additional: ctxt.tcx.def_span(id.owner_id.to_def_id()),
                });
            }
        }
    }
}

fn configure_main(tcx: TyCtxt<'_>, visitor: &EntryContext<'_>) -> Option<(DefId, EntryFnType)> {
    if let Some((local_def_id, _)) = visitor.rustc_main_fn {
        let def_id = local_def_id.to_def_id();
        Some((def_id, EntryFnType::Main { sigpipe: sigpipe(tcx) }))
    } else {
        // The actual resolution of main happens in the resolver, this here
        if let Some(main_def) = tcx.resolutions(()).main_def
            && let Some(def_id) = main_def.opt_fn_def_id()
        {
            // non-local main imports are handled below
            if let Some(def_id) = def_id.as_local()
                && matches!(tcx.hir_node_by_def_id(def_id), Node::ForeignItem(_))
            {
                tcx.dcx().emit_err(ExternMain { span: tcx.def_span(def_id) });
                return None;
            }

            return Some((def_id, EntryFnType::Main { sigpipe: sigpipe(tcx) }));
        }
        no_main_err(tcx, visitor);
        None
    }
}

fn sigpipe(tcx: TyCtxt<'_>) -> u8 {
    match tcx.sess.opts.unstable_opts.on_broken_pipe {
        rustc_target::spec::OnBrokenPipe::Default => sigpipe::DEFAULT,
        rustc_target::spec::OnBrokenPipe::Kill => sigpipe::SIG_DFL,
        rustc_target::spec::OnBrokenPipe::Error => sigpipe::SIG_IGN,
        rustc_target::spec::OnBrokenPipe::Inherit => sigpipe::INHERIT,
    }
}

fn no_main_err(tcx: TyCtxt<'_>, visitor: &EntryContext<'_>) {
    let sp = tcx.def_span(CRATE_DEF_ID);

    // There is no main function.
    let mut has_filename = true;
    let filename = tcx
        .sess
        .local_crate_source_file()
        .map(|src| src.for_scope(&tcx.sess, RemapPathScopeComponents::DIAGNOSTICS).to_path_buf())
        .unwrap_or_else(|| {
            has_filename = false;
            Default::default()
        });
    let main_def_opt = tcx.resolutions(()).main_def;
    let code = E0601;
    let add_teach_note = tcx.sess.teach(code);
    // The file may be empty, which leads to the diagnostic machinery not emitting this
    // note. This is a relatively simple way to detect that case and emit a span-less
    // note instead.
    let file_empty = tcx.sess.source_map().lookup_line(sp.hi()).is_err();

    tcx.dcx().emit_err(NoMainErr {
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
