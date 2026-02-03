use std::path::{Path, PathBuf};

use rustc_macros::Diagnostic;
use rustc_span::{Ident, Span, Symbol};

#[derive(Diagnostic)]
#[diag("unrecognized `DepNode` variant: {$name}")]
pub(crate) struct UnrecognizedDepNode {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag("missing `DepNode` variant")]
pub(crate) struct MissingDepNode {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("no `#[rustc_if_this_changed]` annotation detected")]
pub(crate) struct MissingIfThisChanged {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("OK")]
pub(crate) struct Ok {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("no path from `{$source}` to `{$target}`")]
pub(crate) struct NoPath {
    #[primary_span]
    pub span: Span,
    pub target: Symbol,
    pub source: String,
}

#[derive(Diagnostic)]
#[diag("`except` specified DepNodes that can not be affected for \"{$name}\": \"{$e}\"")]
pub(crate) struct AssertionAuto<'a> {
    #[primary_span]
    pub span: Span,
    pub name: &'a str,
    pub e: &'a str,
}

#[derive(Diagnostic)]
#[diag("clean/dirty auto-assertions not yet defined for Node::Item.node={$kind}")]
pub(crate) struct UndefinedCleanDirtyItem {
    #[primary_span]
    pub span: Span,
    pub kind: String,
}

#[derive(Diagnostic)]
#[diag("clean/dirty auto-assertions not yet defined for {$kind}")]
pub(crate) struct UndefinedCleanDirty {
    #[primary_span]
    pub span: Span,
    pub kind: String,
}

#[derive(Diagnostic)]
#[diag("dep-node label `{$label}` is repeated")]
pub(crate) struct RepeatedDepNodeLabel<'a> {
    #[primary_span]
    pub span: Span,
    pub label: &'a str,
}

#[derive(Diagnostic)]
#[diag("dep-node label `{$label}` not recognized")]
pub(crate) struct UnrecognizedDepNodeLabel<'a> {
    #[primary_span]
    pub span: Span,
    pub label: &'a str,
}

#[derive(Diagnostic)]
#[diag("`{$dep_node_str}` should be dirty but is not")]
pub(crate) struct NotDirty<'a> {
    #[primary_span]
    pub span: Span,
    pub dep_node_str: &'a str,
}

#[derive(Diagnostic)]
#[diag("`{$dep_node_str}` should be clean but is not")]
pub(crate) struct NotClean<'a> {
    #[primary_span]
    pub span: Span,
    pub dep_node_str: &'a str,
}

#[derive(Diagnostic)]
#[diag("`{$dep_node_str}` should have been loaded from disk but it was not")]
pub(crate) struct NotLoaded<'a> {
    #[primary_span]
    pub span: Span,
    pub dep_node_str: &'a str,
}

#[derive(Diagnostic)]
#[diag("unknown `rustc_clean` argument")]
pub(crate) struct UnknownRustcCleanArgument {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("no cfg attribute")]
pub(crate) struct NoCfg {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("associated value expected for `{$ident}`")]
pub(crate) struct AssociatedValueExpectedFor {
    #[primary_span]
    pub span: Span,
    pub ident: Ident,
}

#[derive(Diagnostic)]
#[diag("expected an associated value")]
pub(crate) struct AssociatedValueExpected {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("found unchecked `#[rustc_clean]` attribute")]
pub(crate) struct UncheckedClean {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unable to delete old {$name} at `{$path}`: {$err}")]
pub(crate) struct DeleteOld<'a> {
    pub name: &'a str,
    pub path: PathBuf,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("failed to create {$name} at `{$path}`: {$err}")]
pub(crate) struct CreateNew<'a> {
    pub name: &'a str,
    pub path: PathBuf,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("failed to write {$name} to `{$path}`: {$err}")]
pub(crate) struct WriteNew<'a> {
    pub name: &'a str,
    pub path: PathBuf,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("incremental compilation: error canonicalizing path `{$path}`: {$err}")]
pub(crate) struct CanonicalizePath {
    pub path: PathBuf,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("could not create incremental compilation {$tag} directory `{$path}`: {$err}")]
pub(crate) struct CreateIncrCompDir<'a> {
    pub tag: &'a str,
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("incremental compilation: could not create session directory lock file: {$lock_err}")]
pub(crate) struct CreateLock<'a> {
    pub lock_err: std::io::Error,
    pub session_dir: &'a Path,
    #[note(
        "the filesystem for the incremental path at {$session_dir} does not appear to support locking, consider changing the incremental path to a filesystem that supports locking or disable incremental compilation"
    )]
    pub is_unsupported_lock: bool,
    #[help(
        "incremental compilation can be disabled by setting the environment variable CARGO_INCREMENTAL=0 (see https://doc.rust-lang.org/cargo/reference/profiles.html#incremental)"
    )]
    #[help(
        "the entire build directory can be changed to a different filesystem by setting the environment variable CARGO_TARGET_DIR to a different path (see https://doc.rust-lang.org/cargo/reference/config.html#buildtarget-dir)"
    )]
    pub is_cargo: bool,
}

#[derive(Diagnostic)]
#[diag("error deleting lock file for incremental compilation session directory `{$path}`: {$err}")]
pub(crate) struct DeleteLock<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(
    "hard linking files in the incremental compilation cache failed. copying files instead. consider moving the cache directory to a file system which supports hard linking in session dir `{$path}`"
)]
pub(crate) struct HardLinkFailed<'a> {
    pub path: &'a Path,
}

#[derive(Diagnostic)]
#[diag("failed to delete partly initialized session dir `{$path}`: {$err}")]
pub(crate) struct DeletePartial<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("error deleting incremental compilation session directory `{$path}`: {$err}")]
pub(crate) struct DeleteFull<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("error finalizing incremental compilation session directory `{$path}`: {$err}")]
pub(crate) struct Finalize<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(
    "failed to garbage collect invalid incremental compilation session directory `{$path}`: {$err}"
)]
pub(crate) struct InvalidGcFailed<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(
    "failed to garbage collect finalized incremental compilation session directory `{$path}`: {$err}"
)]
pub(crate) struct FinalizedGcFailed<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("failed to garbage collect incremental compilation session directory `{$path}`: {$err}")]
pub(crate) struct SessionGcFailed<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("we asserted that the incremental cache should not be loaded, but it was loaded")]
pub(crate) struct AssertNotLoaded;

#[derive(Diagnostic)]
#[diag(
    "we asserted that an existing incremental cache directory should be successfully loaded, but it was not"
)]
pub(crate) struct AssertLoaded;

#[derive(Diagnostic)]
#[diag(
    "failed to delete invalidated or incompatible incremental compilation session directory contents `{$path}`: {$err}"
)]
pub(crate) struct DeleteIncompatible {
    pub path: PathBuf,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("could not load dep-graph from `{$path}`: {$err}")]
pub(crate) struct LoadDepGraph {
    pub path: PathBuf,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("failed to move dependency graph from `{$from}` to `{$to}`: {$err}")]
pub(crate) struct MoveDepGraph<'a> {
    pub from: &'a Path,
    pub to: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("failed to create dependency graph at `{$path}`: {$err}")]
pub(crate) struct CreateDepGraph<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("error copying object file `{$from}` to incremental directory as `{$to}`: {$err}")]
pub(crate) struct CopyWorkProductToCache<'a> {
    pub from: &'a Path,
    pub to: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("file-system error deleting outdated file `{$path}`: {$err}")]
pub(crate) struct DeleteWorkProduct<'a> {
    pub path: &'a Path,
    pub err: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(
    "corrupt incremental compilation artifact found at `{$path}`. This file will automatically be ignored and deleted. If you see this message repeatedly or can provoke it without manually manipulating the compiler's artifacts, please file an issue. The incremental compilation system relies on hardlinks and filesystem locks behaving correctly, and may not deal well with OS crashes, so whatever information you can provide about your filesystem or other state may be very relevant."
)]
pub(crate) struct CorruptFile<'a> {
    pub path: &'a Path,
}
