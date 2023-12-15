use rustc_macros::Diagnostic;
use rustc_span::{symbol::Ident, Span, Symbol};
use std::path::{Path, PathBuf};

#[derive(Diagnostic)]
#[diag(incremental_unrecognized_depnode)]
#[must_use]
pub struct UnrecognizedDepNode {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(incremental_missing_depnode)]
#[must_use]
pub struct MissingDepNode {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(incremental_missing_if_this_changed)]
#[must_use]
pub struct MissingIfThisChanged {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(incremental_ok)]
#[must_use]
pub struct Ok {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(incremental_no_path)]
#[must_use]
pub struct NoPath {
    #[primary_span]
    pub span: Span,
    pub target: Symbol,
    pub source: String,
}

#[derive(Diagnostic)]
#[diag(incremental_assertion_auto)]
#[must_use]
pub struct AssertionAuto<'a> {
    #[primary_span]
    pub span: Span,
    pub name: &'a str,
    pub e: &'a str,
}

#[derive(Diagnostic)]
#[diag(incremental_undefined_clean_dirty_assertions_item)]
#[must_use]
pub struct UndefinedCleanDirtyItem {
    #[primary_span]
    pub span: Span,
    pub kind: String,
}

#[derive(Diagnostic)]
#[diag(incremental_undefined_clean_dirty_assertions)]
#[must_use]
pub struct UndefinedCleanDirty {
    #[primary_span]
    pub span: Span,
    pub kind: String,
}

#[derive(Diagnostic)]
#[diag(incremental_repeated_depnode_label)]
#[must_use]
pub struct RepeatedDepNodeLabel<'a> {
    #[primary_span]
    pub span: Span,
    pub label: &'a str,
}

#[derive(Diagnostic)]
#[diag(incremental_unrecognized_depnode_label)]
#[must_use]
pub struct UnrecognizedDepNodeLabel<'a> {
    #[primary_span]
    pub span: Span,
    pub label: &'a str,
}

#[derive(Diagnostic)]
#[diag(incremental_not_dirty)]
#[must_use]
pub struct NotDirty<'a> {
    #[primary_span]
    pub span: Span,
    pub dep_node_str: &'a str,
}

#[derive(Diagnostic)]
#[diag(incremental_not_clean)]
#[must_use]
pub struct NotClean<'a> {
    #[primary_span]
    pub span: Span,
    pub dep_node_str: &'a str,
}

#[derive(Diagnostic)]
#[diag(incremental_not_loaded)]
#[must_use]
pub struct NotLoaded<'a> {
    #[primary_span]
    pub span: Span,
    pub dep_node_str: &'a str,
}

#[derive(Diagnostic)]
#[diag(incremental_unknown_item)]
#[must_use]
pub struct UnknownItem {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(incremental_no_cfg)]
#[must_use]
pub struct NoCfg {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(incremental_associated_value_expected_for)]
#[must_use]
pub struct AssociatedValueExpectedFor {
    #[primary_span]
    pub span: Span,
    pub ident: Ident,
}

#[derive(Diagnostic)]
#[diag(incremental_associated_value_expected)]
#[must_use]
pub struct AssociatedValueExpected {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(incremental_unchecked_clean)]
#[must_use]
pub struct UncheckedClean {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(incremental_delete_old)]
#[must_use]
pub struct DeleteOld<'a> {
    pub name: &'a str,
    pub path: PathBuf,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_create_new)]
#[must_use]
pub struct CreateNew<'a> {
    pub name: &'a str,
    pub path: PathBuf,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_write_new)]
#[must_use]
pub struct WriteNew<'a> {
    pub name: &'a str,
    pub path: PathBuf,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_canonicalize_path)]
#[must_use]
pub struct CanonicalizePath {
    pub path: PathBuf,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_create_incr_comp_dir)]
#[must_use]
pub struct CreateIncrCompDir<'a> {
    pub tag: &'a str,
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_create_lock)]
#[must_use]
pub struct CreateLock<'a> {
    pub lock_err: std::io::Error,
    pub session_dir: &'a Path,
    #[note(incremental_lock_unsupported)]
    pub is_unsupported_lock: Option<()>,
    #[help(incremental_cargo_help_1)]
    #[help(incremental_cargo_help_2)]
    pub is_cargo: Option<()>,
}

#[derive(Diagnostic)]
#[diag(incremental_delete_lock)]
#[must_use]
pub struct DeleteLock<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_hard_link_failed)]
#[must_use]
pub struct HardLinkFailed<'a> {
    pub path: &'a Path,
}

#[derive(Diagnostic)]
#[diag(incremental_delete_partial)]
#[must_use]
pub struct DeletePartial<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_delete_full)]
#[must_use]
pub struct DeleteFull<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_finalize)]
#[must_use]
pub struct Finalize<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_invalid_gc_failed)]
#[must_use]
pub struct InvalidGcFailed<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_finalized_gc_failed)]
#[must_use]
pub struct FinalizedGcFailed<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_session_gc_failed)]
#[must_use]
pub struct SessionGcFailed<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_assert_not_loaded)]
#[must_use]
pub struct AssertNotLoaded;

#[derive(Diagnostic)]
#[diag(incremental_assert_loaded)]
#[must_use]
pub struct AssertLoaded;

#[derive(Diagnostic)]
#[diag(incremental_delete_incompatible)]
#[must_use]
pub struct DeleteIncompatible {
    pub path: PathBuf,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_load_dep_graph)]
#[must_use]
pub struct LoadDepGraph {
    pub path: PathBuf,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_write_dep_graph)]
#[must_use]
pub struct WriteDepGraph<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_move_dep_graph)]
#[must_use]
pub struct MoveDepGraph<'a> {
    pub from: &'a Path,
    pub to: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_create_dep_graph)]
#[must_use]
pub struct CreateDepGraph<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_copy_workproduct_to_cache)]
#[must_use]
pub struct CopyWorkProductToCache<'a> {
    pub from: &'a Path,
    pub to: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_delete_workproduct)]
#[must_use]
pub struct DeleteWorkProduct<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}
