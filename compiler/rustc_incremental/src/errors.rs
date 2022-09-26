use rustc_macros::Diagnostic;
use rustc_span::{symbol::Ident, Span, Symbol};
use std::path::{Path, PathBuf};

#[derive(Diagnostic)]
#[diag(incremental_unrecognized_depnode)]
pub struct UnrecognizedDepNode {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(incremental_missing_depnode)]
pub struct MissingDepNode {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(incremental_missing_if_this_changed)]
pub struct MissingIfThisChanged {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(incremental_ok)]
pub struct Ok {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(incremental_no_path)]
pub struct NoPath {
    #[primary_span]
    pub span: Span,
    pub target: Symbol,
    pub source: String,
}

#[derive(Diagnostic)]
#[diag(incremental_unknown_reuse_kind)]
pub struct UnknownReuseKind {
    #[primary_span]
    pub span: Span,
    pub kind: Symbol,
}

#[derive(Diagnostic)]
#[diag(incremental_missing_query_depgraph)]
pub struct MissingQueryDepGraph {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(incremental_malformed_cgu_name)]
pub struct MalformedCguName {
    #[primary_span]
    pub span: Span,
    pub user_path: String,
    pub crate_name: String,
}

#[derive(Diagnostic)]
#[diag(incremental_no_module_named)]
pub struct NoModuleNamed<'a> {
    #[primary_span]
    pub span: Span,
    pub user_path: &'a str,
    pub cgu_name: Symbol,
    pub cgu_names: String,
}

#[derive(Diagnostic)]
#[diag(incremental_field_associated_value_expected)]
pub struct FieldAssociatedValueExpected {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(incremental_no_field)]
pub struct NoField {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(incremental_assertion_auto)]
pub struct AssertionAuto<'a> {
    #[primary_span]
    pub span: Span,
    pub name: &'a str,
    pub e: &'a str,
}

#[derive(Diagnostic)]
#[diag(incremental_undefined_clean_dirty_assertions_item)]
pub struct UndefinedCleanDirtyItem {
    #[primary_span]
    pub span: Span,
    pub kind: String,
}

#[derive(Diagnostic)]
#[diag(incremental_undefined_clean_dirty_assertions)]
pub struct UndefinedCleanDirty {
    #[primary_span]
    pub span: Span,
    pub kind: String,
}

#[derive(Diagnostic)]
#[diag(incremental_repeated_depnode_label)]
pub struct RepeatedDepNodeLabel<'a> {
    #[primary_span]
    pub span: Span,
    pub label: &'a str,
}

#[derive(Diagnostic)]
#[diag(incremental_unrecognized_depnode_label)]
pub struct UnrecognizedDepNodeLabel<'a> {
    #[primary_span]
    pub span: Span,
    pub label: &'a str,
}

#[derive(Diagnostic)]
#[diag(incremental_not_dirty)]
pub struct NotDirty<'a> {
    #[primary_span]
    pub span: Span,
    pub dep_node_str: &'a str,
}

#[derive(Diagnostic)]
#[diag(incremental_not_clean)]
pub struct NotClean<'a> {
    #[primary_span]
    pub span: Span,
    pub dep_node_str: &'a str,
}

#[derive(Diagnostic)]
#[diag(incremental_not_loaded)]
pub struct NotLoaded<'a> {
    #[primary_span]
    pub span: Span,
    pub dep_node_str: &'a str,
}

#[derive(Diagnostic)]
#[diag(incremental_unknown_item)]
pub struct UnknownItem {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(incremental_no_cfg)]
pub struct NoCfg {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(incremental_associated_value_expected_for)]
pub struct AssociatedValueExpectedFor {
    #[primary_span]
    pub span: Span,
    pub ident: Ident,
}

#[derive(Diagnostic)]
#[diag(incremental_associated_value_expected)]
pub struct AssociatedValueExpected {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(incremental_unchecked_clean)]
pub struct UncheckedClean {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(incremental_delete_old)]
pub struct DeleteOld<'a> {
    pub name: &'a str,
    pub path: PathBuf,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_create_new)]
pub struct CreateNew<'a> {
    pub name: &'a str,
    pub path: PathBuf,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_write_new)]
pub struct WriteNew<'a> {
    pub name: &'a str,
    pub path: PathBuf,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_canonicalize_path)]
pub struct CanonicalizePath {
    pub path: PathBuf,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_create_incr_comp_dir)]
pub struct CreateIncrCompDir<'a> {
    pub tag: &'a str,
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_create_lock)]
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
pub struct DeleteLock<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_hard_link_failed)]
pub struct HardLinkFailed<'a> {
    pub path: &'a Path,
}

#[derive(Diagnostic)]
#[diag(incremental_delete_partial)]
pub struct DeletePartial<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_delete_full)]
pub struct DeleteFull<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_finalize)]
pub struct Finalize<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_invalid_gc_failed)]
pub struct InvalidGcFailed<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_finalized_gc_failed)]
pub struct FinalizedGcFailed<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_session_gc_failed)]
pub struct SessionGcFailed<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_assert_not_loaded)]
pub struct AssertNotLoaded;

#[derive(Diagnostic)]
#[diag(incremental_assert_loaded)]
pub struct AssertLoaded;

#[derive(Diagnostic)]
#[diag(incremental_delete_incompatible)]
pub struct DeleteIncompatible {
    pub path: PathBuf,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_load_dep_graph)]
pub struct LoadDepGraph {
    pub path: PathBuf,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_decode_incr_cache)]
pub struct DecodeIncrCache {
    pub err: String,
}

#[derive(Diagnostic)]
#[diag(incremental_write_dep_graph)]
pub struct WriteDepGraph<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_move_dep_graph)]
pub struct MoveDepGraph<'a> {
    pub from: &'a Path,
    pub to: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_create_dep_graph)]
pub struct CreateDepGraph<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_copy_workproduct_to_cache)]
pub struct CopyWorkProductToCache<'a> {
    pub from: &'a Path,
    pub to: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(incremental_delete_workproduct)]
pub struct DeleteWorkProduct<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}
