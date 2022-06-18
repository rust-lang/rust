use rustc_ast::entry::EntryPointType;
use rustc_errors::struct_span_err;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId, CRATE_DEF_ID, LOCAL_CRATE};
use rustc_hir::{ItemId, Node, CRATE_HIR_ID};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{DefIdTree, TyCtxt};
use rustc_session::config::{CrateType, EntryFnType};
use rustc_session::parse::feature_err;
use rustc_session::Session;
use rustc_span::symbol::sym;
use rustc_span::{Span, DUMMY_SP};

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
        EntryPointType::MainAttr
    } else {
        if let Some(name) = ctxt.tcx.opt_item_name(id.def_id.to_def_id())
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

fn throw_attr_err(sess: &Session, span: Span, attr: &str) {
    sess.struct_span_err(span, &format!("`{}` attribute can only be used on functions", attr))
        .emit();
}

fn find_item(id: ItemId, ctxt: &mut EntryContext<'_>) {
    let at_root = ctxt.tcx.opt_local_parent(id.def_id) == Some(CRATE_DEF_ID);

    match entry_point_type(ctxt, id, at_root) {
        EntryPointType::None => (),
        _ if !matches!(ctxt.tcx.def_kind(id.def_id), DefKind::Fn) => {
            let attrs = ctxt.tcx.hir().attrs(id.hir_id());
            if let Some(attr) = ctxt.tcx.sess.find_by_name(attrs, sym::start) {
                throw_attr_err(&ctxt.tcx.sess, attr.span, "start");
            }
            if let Some(attr) = ctxt.tcx.sess.find_by_name(attrs, sym::rustc_main) {
                throw_attr_err(&ctxt.tcx.sess, attr.span, "rustc_main");
            }
        }
        EntryPointType::MainNamed => (),
        EntryPointType::OtherMain => {
            ctxt.non_main_fns.push(ctxt.tcx.def_span(id.def_id));
        }
        EntryPointType::MainAttr => {
            if ctxt.attr_main_fn.is_none() {
                ctxt.attr_main_fn = Some((id.def_id, ctxt.tcx.def_span(id.def_id.to_def_id())));
            } else {
                struct_span_err!(
                    ctxt.tcx.sess,
                    ctxt.tcx.def_span(id.def_id.to_def_id()),
                    E0137,
                    "multiple functions with a `#[main]` attribute"
                )
                .span_label(
                    ctxt.tcx.def_span(id.def_id.to_def_id()),
                    "additional `#[main]` function",
                )
                .span_label(ctxt.attr_main_fn.unwrap().1, "first `#[main]` function")
                .emit();
            }
        }
        EntryPointType::Start => {
            if ctxt.start_fn.is_none() {
                ctxt.start_fn = Some((id.def_id, ctxt.tcx.def_span(id.def_id.to_def_id())));
            } else {
                struct_span_err!(
                    ctxt.tcx.sess,
                    ctxt.tcx.def_span(id.def_id.to_def_id()),
                    E0138,
                    "multiple `start` functions"
                )
                .span_label(ctxt.start_fn.unwrap().1, "previous `#[start]` function here")
                .span_label(ctxt.tcx.def_span(id.def_id.to_def_id()), "multiple `start` functions")
                .emit();
            }
        }
    }
}

fn configure_main(tcx: TyCtxt<'_>, visitor: &EntryContext<'_>) -> Option<(DefId, EntryFnType)> {
    if let Some((def_id, _)) = visitor.start_fn {
        Some((def_id.to_def_id(), EntryFnType::Start))
    } else if let Some((def_id, _)) = visitor.attr_main_fn {
        Some((def_id.to_def_id(), EntryFnType::Main))
    } else {
        if let Some(main_def) = tcx.resolutions(()).main_def && let Some(def_id) = main_def.opt_fn_def_id() {
            // non-local main imports are handled below
            if let Some(def_id) = def_id.as_local() && matches!(tcx.hir().find_by_def_id(def_id), Some(Node::ForeignItem(_))) {
                tcx.sess
                    .struct_span_err(
                        tcx.def_span(def_id),
                        "the `main` function cannot be declared in an `extern` block",
                    )
                    .emit();
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
            return Some((def_id, EntryFnType::Main));
        }
        no_main_err(tcx, visitor);
        None
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
    let mut err = struct_span_err!(
        tcx.sess,
        DUMMY_SP,
        E0601,
        "`main` function not found in crate `{}`",
        tcx.crate_name(LOCAL_CRATE)
    );
    let filename = &tcx.sess.local_crate_source_file;
    let note = if !visitor.non_main_fns.is_empty() {
        for &span in &visitor.non_main_fns {
            err.span_note(span, "here is a function named `main`");
        }
        err.note("you have one or more functions named `main` not defined at the crate level");
        err.help("consider moving the `main` function definitions");
        // There were some functions named `main` though. Try to give the user a hint.
        format!(
            "the main function must be defined at the crate level{}",
            filename.as_ref().map(|f| format!(" (in `{}`)", f.display())).unwrap_or_default()
        )
    } else if let Some(filename) = filename {
        format!("consider adding a `main` function to `{}`", filename.display())
    } else {
        String::from("consider adding a `main` function at the crate level")
    };
    // The file may be empty, which leads to the diagnostic machinery not emitting this
    // note. This is a relatively simple way to detect that case and emit a span-less
    // note instead.
    if tcx.sess.source_map().lookup_line(sp.hi()).is_ok() {
        err.set_span(sp.shrink_to_hi());
        err.span_label(sp.shrink_to_hi(), &note);
    } else {
        err.note(&note);
    }

    if let Some(main_def) = tcx.resolutions(()).main_def && main_def.opt_fn_def_id().is_none(){
        // There is something at `crate::main`, but it is not a function definition.
        err.span_label(main_def.span, "non-function item at `crate::main` is found");
    }

    if tcx.sess.teach(&err.get_code().unwrap()) {
        err.note(
            "If you don't know the basics of Rust, you can go look to the Rust Book \
                  to get started: https://doc.rust-lang.org/book/",
        );
    }
    err.emit();
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers { entry_fn, ..*providers };
}
