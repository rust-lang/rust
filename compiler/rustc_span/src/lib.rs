//! Source positions and related helper functions.
//!
//! Important concepts in this module include:
//!
//! - the *span*, represented by [`SpanData`] and related types;
//! - source code as represented by a [`SourceMap`]; and
//! - interned strings, represented by [`Symbol`]s, with some common symbols available statically in the [`sym`] module.
//!
//! Unlike most compilers, the span contains not only the position in the source code, but also various other metadata,
//! such as the edition and macro hygiene. This metadata is stored in [`SyntaxContext`] and [`ExpnData`].
//!
//! ## Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(array_windows)]
#![feature(if_let_guard)]
#![feature(negative_impls)]
#![feature(min_specialization)]
#![feature(rustc_attrs)]
#![feature(let_chains)]
#![feature(round_char_boundary)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate rustc_macros;

#[macro_use]
extern crate tracing;

use rustc_data_structures::AtomicRef;
use rustc_macros::HashStable_Generic;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};

mod caching_source_map_view;
pub mod source_map;
pub use self::caching_source_map_view::CachingSourceMapView;
use source_map::SourceMap;

pub mod edition;
use edition::Edition;
pub mod hygiene;
use hygiene::Transparency;
pub use hygiene::{DesugaringKind, ExpnKind, MacroKind};
pub use hygiene::{ExpnData, ExpnHash, ExpnId, LocalExpnId, SyntaxContext};
use rustc_data_structures::stable_hasher::HashingControls;
pub mod def_id;
use def_id::{CrateNum, DefId, DefPathHash, LocalDefId, LOCAL_CRATE};
pub mod edit_distance;
mod span_encoding;
pub use span_encoding::{Span, DUMMY_SP};

pub mod symbol;
pub use symbol::{sym, Symbol};

mod analyze_source_file;
pub mod fatal_error;

pub mod profiling;

use rustc_data_structures::stable_hasher::{Hash128, Hash64, HashStable, StableHasher};
use rustc_data_structures::sync::{Lock, Lrc};

use std::borrow::Cow;
use std::cmp::{self, Ordering};
use std::hash::Hash;
use std::ops::{Add, Range, Sub};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::{fmt, iter};

use md5::Digest;
use md5::Md5;
use sha1::Sha1;
use sha2::Sha256;

#[cfg(test)]
mod tests;

/// Per-session global variables: this struct is stored in thread-local storage
/// in such a way that it is accessible without any kind of handle to all
/// threads within the compilation session, but is not accessible outside the
/// session.
pub struct SessionGlobals {
    symbol_interner: symbol::Interner,
    span_interner: Lock<span_encoding::SpanInterner>,
    hygiene_data: Lock<hygiene::HygieneData>,

    /// A reference to the source map in the `Session`. It's an `Option`
    /// because it can't be initialized until `Session` is created, which
    /// happens after `SessionGlobals`. `set_source_map` does the
    /// initialization.
    ///
    /// This field should only be used in places where the `Session` is truly
    /// not available, such as `<Span as Debug>::fmt`.
    source_map: Lock<Option<Lrc<SourceMap>>>,
}

impl SessionGlobals {
    pub fn new(edition: Edition) -> SessionGlobals {
        SessionGlobals {
            symbol_interner: symbol::Interner::fresh(),
            span_interner: Lock::new(span_encoding::SpanInterner::default()),
            hygiene_data: Lock::new(hygiene::HygieneData::new(edition)),
            source_map: Lock::new(None),
        }
    }
}

#[inline]
pub fn create_session_globals_then<R>(edition: Edition, f: impl FnOnce() -> R) -> R {
    assert!(
        !SESSION_GLOBALS.is_set(),
        "SESSION_GLOBALS should never be overwritten! \
         Use another thread if you need another SessionGlobals"
    );
    let session_globals = SessionGlobals::new(edition);
    SESSION_GLOBALS.set(&session_globals, f)
}

#[inline]
pub fn set_session_globals_then<R>(session_globals: &SessionGlobals, f: impl FnOnce() -> R) -> R {
    assert!(
        !SESSION_GLOBALS.is_set(),
        "SESSION_GLOBALS should never be overwritten! \
         Use another thread if you need another SessionGlobals"
    );
    SESSION_GLOBALS.set(session_globals, f)
}

#[inline]
pub fn create_default_session_if_not_set_then<R, F>(f: F) -> R
where
    F: FnOnce(&SessionGlobals) -> R,
{
    create_session_if_not_set_then(edition::DEFAULT_EDITION, f)
}

#[inline]
pub fn create_session_if_not_set_then<R, F>(edition: Edition, f: F) -> R
where
    F: FnOnce(&SessionGlobals) -> R,
{
    if !SESSION_GLOBALS.is_set() {
        let session_globals = SessionGlobals::new(edition);
        SESSION_GLOBALS.set(&session_globals, || SESSION_GLOBALS.with(f))
    } else {
        SESSION_GLOBALS.with(f)
    }
}

#[inline]
pub fn with_session_globals<R, F>(f: F) -> R
where
    F: FnOnce(&SessionGlobals) -> R,
{
    SESSION_GLOBALS.with(f)
}

#[inline]
pub fn create_default_session_globals_then<R>(f: impl FnOnce() -> R) -> R {
    create_session_globals_then(edition::DEFAULT_EDITION, f)
}

// If this ever becomes non thread-local, `decode_syntax_context`
// and `decode_expn_id` will need to be updated to handle concurrent
// deserialization.
scoped_tls::scoped_thread_local!(static SESSION_GLOBALS: SessionGlobals);

// FIXME: We should use this enum or something like it to get rid of the
// use of magic `/rust/1.x/...` paths across the board.
#[derive(Debug, Eq, PartialEq, Clone, Ord, PartialOrd)]
#[derive(Decodable)]
pub enum RealFileName {
    LocalPath(PathBuf),
    /// For remapped paths (namely paths into libstd that have been mapped
    /// to the appropriate spot on the local host's file system, and local file
    /// system paths that have been remapped with `FilePathMapping`),
    Remapped {
        /// `local_path` is the (host-dependent) local path to the file. This is
        /// None if the file was imported from another crate
        local_path: Option<PathBuf>,
        /// `virtual_name` is the stable path rustc will store internally within
        /// build artifacts.
        virtual_name: PathBuf,
    },
}

impl Hash for RealFileName {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // To prevent #70924 from happening again we should only hash the
        // remapped (virtualized) path if that exists. This is because
        // virtualized paths to sysroot crates (/rust/$hash or /rust/$version)
        // remain stable even if the corresponding local_path changes
        self.remapped_path_if_available().hash(state)
    }
}

// This is functionally identical to #[derive(Encodable)], with the exception of
// an added assert statement
impl<S: Encoder> Encodable<S> for RealFileName {
    fn encode(&self, encoder: &mut S) {
        match *self {
            RealFileName::LocalPath(ref local_path) => encoder.emit_enum_variant(0, |encoder| {
                local_path.encode(encoder);
            }),

            RealFileName::Remapped { ref local_path, ref virtual_name } => encoder
                .emit_enum_variant(1, |encoder| {
                    // For privacy and build reproducibility, we must not embed host-dependant path in artifacts
                    // if they have been remapped by --remap-path-prefix
                    assert!(local_path.is_none());
                    local_path.encode(encoder);
                    virtual_name.encode(encoder);
                }),
        }
    }
}

impl RealFileName {
    /// Returns the path suitable for reading from the file system on the local host,
    /// if this information exists.
    /// Avoid embedding this in build artifacts; see `remapped_path_if_available()` for that.
    pub fn local_path(&self) -> Option<&Path> {
        match self {
            RealFileName::LocalPath(p) => Some(p),
            RealFileName::Remapped { local_path, virtual_name: _ } => local_path.as_deref(),
        }
    }

    /// Returns the path suitable for reading from the file system on the local host,
    /// if this information exists.
    /// Avoid embedding this in build artifacts; see `remapped_path_if_available()` for that.
    pub fn into_local_path(self) -> Option<PathBuf> {
        match self {
            RealFileName::LocalPath(p) => Some(p),
            RealFileName::Remapped { local_path: p, virtual_name: _ } => p,
        }
    }

    /// Returns the path suitable for embedding into build artifacts. This would still
    /// be a local path if it has not been remapped. A remapped path will not correspond
    /// to a valid file system path: see `local_path_if_available()` for something that
    /// is more likely to return paths into the local host file system.
    pub fn remapped_path_if_available(&self) -> &Path {
        match self {
            RealFileName::LocalPath(p)
            | RealFileName::Remapped { local_path: _, virtual_name: p } => p,
        }
    }

    /// Returns the path suitable for reading from the file system on the local host,
    /// if this information exists. Otherwise returns the remapped name.
    /// Avoid embedding this in build artifacts; see `remapped_path_if_available()` for that.
    pub fn local_path_if_available(&self) -> &Path {
        match self {
            RealFileName::LocalPath(path)
            | RealFileName::Remapped { local_path: None, virtual_name: path }
            | RealFileName::Remapped { local_path: Some(path), virtual_name: _ } => path,
        }
    }

    pub fn to_string_lossy(&self, display_pref: FileNameDisplayPreference) -> Cow<'_, str> {
        match display_pref {
            FileNameDisplayPreference::Local => self.local_path_if_available().to_string_lossy(),
            FileNameDisplayPreference::Remapped => {
                self.remapped_path_if_available().to_string_lossy()
            }
            FileNameDisplayPreference::Short => self
                .local_path_if_available()
                .file_name()
                .map_or_else(|| "".into(), |f| f.to_string_lossy()),
        }
    }
}

/// Differentiates between real files and common virtual files.
#[derive(Debug, Eq, PartialEq, Clone, Ord, PartialOrd, Hash)]
#[derive(Decodable, Encodable)]
pub enum FileName {
    Real(RealFileName),
    /// Call to `quote!`.
    QuoteExpansion(Hash64),
    /// Command line.
    Anon(Hash64),
    /// Hack in `src/librustc_ast/parse.rs`.
    // FIXME(jseyfried)
    MacroExpansion(Hash64),
    ProcMacroSourceCode(Hash64),
    /// Strings provided as `--cfg [cfgspec]` stored in a `crate_cfg`.
    CfgSpec(Hash64),
    /// Strings provided as crate attributes in the CLI.
    CliCrateAttr(Hash64),
    /// Custom sources for explicit parser calls from plugins and drivers.
    Custom(String),
    DocTest(PathBuf, isize),
    /// Post-substitution inline assembly from LLVM.
    InlineAsm(Hash64),
}

impl From<PathBuf> for FileName {
    fn from(p: PathBuf) -> Self {
        assert!(!p.to_string_lossy().ends_with('>'));
        FileName::Real(RealFileName::LocalPath(p))
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub enum FileNameDisplayPreference {
    /// Display the path after the application of rewrite rules provided via `--remap-path-prefix`.
    /// This is appropriate for paths that get embedded into files produced by the compiler.
    Remapped,
    /// Display the path before the application of rewrite rules provided via `--remap-path-prefix`.
    /// This is appropriate for use in user-facing output (such as diagnostics).
    Local,
    /// Display only the filename, as a way to reduce the verbosity of the output.
    /// This is appropriate for use in user-facing output (such as diagnostics).
    Short,
}

pub struct FileNameDisplay<'a> {
    inner: &'a FileName,
    display_pref: FileNameDisplayPreference,
}

impl fmt::Display for FileNameDisplay<'_> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use FileName::*;
        match *self.inner {
            Real(ref name) => {
                write!(fmt, "{}", name.to_string_lossy(self.display_pref))
            }
            QuoteExpansion(_) => write!(fmt, "<quote expansion>"),
            MacroExpansion(_) => write!(fmt, "<macro expansion>"),
            Anon(_) => write!(fmt, "<anon>"),
            ProcMacroSourceCode(_) => write!(fmt, "<proc-macro source code>"),
            CfgSpec(_) => write!(fmt, "<cfgspec>"),
            CliCrateAttr(_) => write!(fmt, "<crate attribute>"),
            Custom(ref s) => write!(fmt, "<{s}>"),
            DocTest(ref path, _) => write!(fmt, "{}", path.display()),
            InlineAsm(_) => write!(fmt, "<inline asm>"),
        }
    }
}

impl<'a> FileNameDisplay<'a> {
    pub fn to_string_lossy(&self) -> Cow<'a, str> {
        match self.inner {
            FileName::Real(ref inner) => inner.to_string_lossy(self.display_pref),
            _ => Cow::from(self.to_string()),
        }
    }
}

impl FileName {
    pub fn is_real(&self) -> bool {
        use FileName::*;
        match *self {
            Real(_) => true,
            Anon(_)
            | MacroExpansion(_)
            | ProcMacroSourceCode(_)
            | CfgSpec(_)
            | CliCrateAttr(_)
            | Custom(_)
            | QuoteExpansion(_)
            | DocTest(_, _)
            | InlineAsm(_) => false,
        }
    }

    pub fn prefer_remapped(&self) -> FileNameDisplay<'_> {
        FileNameDisplay { inner: self, display_pref: FileNameDisplayPreference::Remapped }
    }

    /// This may include transient local filesystem information.
    /// Must not be embedded in build outputs.
    pub fn prefer_local(&self) -> FileNameDisplay<'_> {
        FileNameDisplay { inner: self, display_pref: FileNameDisplayPreference::Local }
    }

    pub fn display(&self, display_pref: FileNameDisplayPreference) -> FileNameDisplay<'_> {
        FileNameDisplay { inner: self, display_pref }
    }

    pub fn macro_expansion_source_code(src: &str) -> FileName {
        let mut hasher = StableHasher::new();
        src.hash(&mut hasher);
        FileName::MacroExpansion(hasher.finish())
    }

    pub fn anon_source_code(src: &str) -> FileName {
        let mut hasher = StableHasher::new();
        src.hash(&mut hasher);
        FileName::Anon(hasher.finish())
    }

    pub fn proc_macro_source_code(src: &str) -> FileName {
        let mut hasher = StableHasher::new();
        src.hash(&mut hasher);
        FileName::ProcMacroSourceCode(hasher.finish())
    }

    pub fn cfg_spec_source_code(src: &str) -> FileName {
        let mut hasher = StableHasher::new();
        src.hash(&mut hasher);
        FileName::QuoteExpansion(hasher.finish())
    }

    pub fn cli_crate_attr_source_code(src: &str) -> FileName {
        let mut hasher = StableHasher::new();
        src.hash(&mut hasher);
        FileName::CliCrateAttr(hasher.finish())
    }

    pub fn doc_test_source_code(path: PathBuf, line: isize) -> FileName {
        FileName::DocTest(path, line)
    }

    pub fn inline_asm_source_code(src: &str) -> FileName {
        let mut hasher = StableHasher::new();
        src.hash(&mut hasher);
        FileName::InlineAsm(hasher.finish())
    }
}

/// Represents a span.
///
/// Spans represent a region of code, used for error reporting. Positions in spans
/// are *absolute* positions from the beginning of the [`SourceMap`], not positions
/// relative to [`SourceFile`]s. Methods on the `SourceMap` can be used to relate spans back
/// to the original source.
///
/// You must be careful if the span crosses more than one file, since you will not be
/// able to use many of the functions on spans in source_map and you cannot assume
/// that the length of the span is equal to `span.hi - span.lo`; there may be space in the
/// [`BytePos`] range between files.
///
/// `SpanData` is public because `Span` uses a thread-local interner and can't be
/// sent to other threads, but some pieces of performance infra run in a separate thread.
/// Using `Span` is generally preferred.
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct SpanData {
    pub lo: BytePos,
    pub hi: BytePos,
    /// Information about where the macro came from, if this piece of
    /// code was created by a macro expansion.
    pub ctxt: SyntaxContext,
    pub parent: Option<LocalDefId>,
}

// Order spans by position in the file.
impl Ord for SpanData {
    fn cmp(&self, other: &Self) -> Ordering {
        let SpanData {
            lo: s_lo,
            hi: s_hi,
            ctxt: s_ctxt,
            // `LocalDefId` does not implement `Ord`.
            // The other fields are enough to determine in-file order.
            parent: _,
        } = self;
        let SpanData {
            lo: o_lo,
            hi: o_hi,
            ctxt: o_ctxt,
            // `LocalDefId` does not implement `Ord`.
            // The other fields are enough to determine in-file order.
            parent: _,
        } = other;

        (s_lo, s_hi, s_ctxt).cmp(&(o_lo, o_hi, o_ctxt))
    }
}

impl PartialOrd for SpanData {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl SpanData {
    #[inline]
    pub fn span(&self) -> Span {
        Span::new(self.lo, self.hi, self.ctxt, self.parent)
    }
    #[inline]
    pub fn with_lo(&self, lo: BytePos) -> Span {
        Span::new(lo, self.hi, self.ctxt, self.parent)
    }
    #[inline]
    pub fn with_hi(&self, hi: BytePos) -> Span {
        Span::new(self.lo, hi, self.ctxt, self.parent)
    }
    #[inline]
    pub fn with_ctxt(&self, ctxt: SyntaxContext) -> Span {
        Span::new(self.lo, self.hi, ctxt, self.parent)
    }
    #[inline]
    pub fn with_parent(&self, parent: Option<LocalDefId>) -> Span {
        Span::new(self.lo, self.hi, self.ctxt, parent)
    }
    /// Returns `true` if this is a dummy span with any hygienic context.
    #[inline]
    pub fn is_dummy(self) -> bool {
        self.lo.0 == 0 && self.hi.0 == 0
    }
    #[inline]
    pub fn is_visible(self, sm: &SourceMap) -> bool {
        !self.is_dummy() && sm.is_span_accessible(self.span())
    }
    /// Returns `true` if `self` fully encloses `other`.
    pub fn contains(self, other: Self) -> bool {
        self.lo <= other.lo && other.hi <= self.hi
    }
}

// The interner is pointed to by a thread local value which is only set on the main thread
// with parallelization is disabled. So we don't allow `Span` to transfer between threads
// to avoid panics and other errors, even though it would be memory safe to do so.
#[cfg(not(parallel_compiler))]
impl !Send for Span {}
#[cfg(not(parallel_compiler))]
impl !Sync for Span {}

impl PartialOrd for Span {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&self.data(), &rhs.data())
    }
}
impl Ord for Span {
    fn cmp(&self, rhs: &Self) -> Ordering {
        Ord::cmp(&self.data(), &rhs.data())
    }
}

impl Span {
    #[inline]
    pub fn lo(self) -> BytePos {
        self.data().lo
    }
    #[inline]
    pub fn with_lo(self, lo: BytePos) -> Span {
        self.data().with_lo(lo)
    }
    #[inline]
    pub fn hi(self) -> BytePos {
        self.data().hi
    }
    #[inline]
    pub fn with_hi(self, hi: BytePos) -> Span {
        self.data().with_hi(hi)
    }
    #[inline]
    pub fn eq_ctxt(self, other: Span) -> bool {
        self.data_untracked().ctxt == other.data_untracked().ctxt
    }
    #[inline]
    pub fn with_ctxt(self, ctxt: SyntaxContext) -> Span {
        self.data_untracked().with_ctxt(ctxt)
    }
    #[inline]
    pub fn parent(self) -> Option<LocalDefId> {
        self.data().parent
    }
    #[inline]
    pub fn with_parent(self, ctxt: Option<LocalDefId>) -> Span {
        self.data().with_parent(ctxt)
    }

    /// Returns `true` if this is a dummy span with any hygienic context.
    #[inline]
    pub fn is_dummy(self) -> bool {
        self.data_untracked().is_dummy()
    }

    #[inline]
    pub fn is_visible(self, sm: &SourceMap) -> bool {
        self.data_untracked().is_visible(sm)
    }

    /// Returns `true` if this span comes from any kind of macro, desugaring or inlining.
    #[inline]
    pub fn from_expansion(self) -> bool {
        self.ctxt() != SyntaxContext::root()
    }

    /// Returns `true` if `span` originates in a macro's expansion where debuginfo should be
    /// collapsed.
    pub fn in_macro_expansion_with_collapse_debuginfo(self) -> bool {
        let outer_expn = self.ctxt().outer_expn_data();
        matches!(outer_expn.kind, ExpnKind::Macro(..)) && outer_expn.collapse_debuginfo
    }

    /// Returns `true` if `span` originates in a derive-macro's expansion.
    pub fn in_derive_expansion(self) -> bool {
        matches!(self.ctxt().outer_expn_data().kind, ExpnKind::Macro(MacroKind::Derive, _))
    }

    /// Gate suggestions that would not be appropriate in a context the user didn't write.
    pub fn can_be_used_for_suggestions(self) -> bool {
        !self.from_expansion()
        // FIXME: If this span comes from a `derive` macro but it points at code the user wrote,
        // the callsite span and the span will be pointing at different places. It also means that
        // we can safely provide suggestions on this span.
            || (matches!(self.ctxt().outer_expn_data().kind, ExpnKind::Macro(MacroKind::Derive, _))
                && self.parent_callsite().map(|p| (p.lo(), p.hi())) != Some((self.lo(), self.hi())))
    }

    #[inline]
    pub fn with_root_ctxt(lo: BytePos, hi: BytePos) -> Span {
        Span::new(lo, hi, SyntaxContext::root(), None)
    }

    /// Returns a new span representing an empty span at the beginning of this span.
    #[inline]
    pub fn shrink_to_lo(self) -> Span {
        let span = self.data_untracked();
        span.with_hi(span.lo)
    }
    /// Returns a new span representing an empty span at the end of this span.
    #[inline]
    pub fn shrink_to_hi(self) -> Span {
        let span = self.data_untracked();
        span.with_lo(span.hi)
    }

    #[inline]
    /// Returns `true` if `hi == lo`.
    pub fn is_empty(self) -> bool {
        let span = self.data_untracked();
        span.hi == span.lo
    }

    /// Returns `self` if `self` is not the dummy span, and `other` otherwise.
    pub fn substitute_dummy(self, other: Span) -> Span {
        if self.is_dummy() { other } else { self }
    }

    /// Returns `true` if `self` fully encloses `other`.
    pub fn contains(self, other: Span) -> bool {
        let span = self.data();
        let other = other.data();
        span.contains(other)
    }

    /// Returns `true` if `self` touches `other`.
    pub fn overlaps(self, other: Span) -> bool {
        let span = self.data();
        let other = other.data();
        span.lo < other.hi && other.lo < span.hi
    }

    /// Returns `true` if the spans are equal with regards to the source text.
    ///
    /// Use this instead of `==` when either span could be generated code,
    /// and you only care that they point to the same bytes of source text.
    pub fn source_equal(self, other: Span) -> bool {
        let span = self.data();
        let other = other.data();
        span.lo == other.lo && span.hi == other.hi
    }

    /// Returns `Some(span)`, where the start is trimmed by the end of `other`.
    pub fn trim_start(self, other: Span) -> Option<Span> {
        let span = self.data();
        let other = other.data();
        if span.hi > other.hi { Some(span.with_lo(cmp::max(span.lo, other.hi))) } else { None }
    }

    /// Returns the source span -- this is either the supplied span, or the span for
    /// the macro callsite that expanded to it.
    pub fn source_callsite(self) -> Span {
        let expn_data = self.ctxt().outer_expn_data();
        if !expn_data.is_root() { expn_data.call_site.source_callsite() } else { self }
    }

    /// The `Span` for the tokens in the previous macro expansion from which `self` was generated,
    /// if any.
    pub fn parent_callsite(self) -> Option<Span> {
        let expn_data = self.ctxt().outer_expn_data();
        if !expn_data.is_root() { Some(expn_data.call_site) } else { None }
    }

    /// Walk down the expansion ancestors to find a span that's contained within `outer`.
    pub fn find_ancestor_inside(mut self, outer: Span) -> Option<Span> {
        while !outer.contains(self) {
            self = self.parent_callsite()?;
        }
        Some(self)
    }

    /// Like `find_ancestor_inside`, but specifically for when spans might not
    /// overlaps. Take care when using this, and prefer `find_ancestor_inside`
    /// when you know that the spans are nested (modulo macro expansion).
    pub fn find_ancestor_in_same_ctxt(mut self, other: Span) -> Option<Span> {
        while !Span::eq_ctxt(self, other) {
            self = self.parent_callsite()?;
        }
        Some(self)
    }

    /// Edition of the crate from which this span came.
    pub fn edition(self) -> edition::Edition {
        self.ctxt().edition()
    }

    #[inline]
    pub fn is_rust_2015(self) -> bool {
        self.edition().is_rust_2015()
    }

    #[inline]
    pub fn rust_2018(self) -> bool {
        self.edition().rust_2018()
    }

    #[inline]
    pub fn rust_2021(self) -> bool {
        self.edition().rust_2021()
    }

    #[inline]
    pub fn rust_2024(self) -> bool {
        self.edition().rust_2024()
    }

    /// Returns the source callee.
    ///
    /// Returns `None` if the supplied span has no expansion trace,
    /// else returns the `ExpnData` for the macro definition
    /// corresponding to the source callsite.
    pub fn source_callee(self) -> Option<ExpnData> {
        let expn_data = self.ctxt().outer_expn_data();

        // Create an iterator of call site expansions
        iter::successors(Some(expn_data), |expn_data| {
            Some(expn_data.call_site.ctxt().outer_expn_data())
        })
        // Find the last expansion which is not root
        .take_while(|expn_data| !expn_data.is_root())
        .last()
    }

    /// Checks if a span is "internal" to a macro in which `#[unstable]`
    /// items can be used (that is, a macro marked with
    /// `#[allow_internal_unstable]`).
    pub fn allows_unstable(self, feature: Symbol) -> bool {
        self.ctxt()
            .outer_expn_data()
            .allow_internal_unstable
            .is_some_and(|features| features.iter().any(|&f| f == feature))
    }

    /// Checks if this span arises from a compiler desugaring of kind `kind`.
    pub fn is_desugaring(self, kind: DesugaringKind) -> bool {
        match self.ctxt().outer_expn_data().kind {
            ExpnKind::Desugaring(k) => k == kind,
            _ => false,
        }
    }

    /// Returns the compiler desugaring that created this span, or `None`
    /// if this span is not from a desugaring.
    pub fn desugaring_kind(self) -> Option<DesugaringKind> {
        match self.ctxt().outer_expn_data().kind {
            ExpnKind::Desugaring(k) => Some(k),
            _ => None,
        }
    }

    /// Checks if a span is "internal" to a macro in which `unsafe`
    /// can be used without triggering the `unsafe_code` lint.
    /// (that is, a macro marked with `#[allow_internal_unsafe]`).
    pub fn allows_unsafe(self) -> bool {
        self.ctxt().outer_expn_data().allow_internal_unsafe
    }

    pub fn macro_backtrace(mut self) -> impl Iterator<Item = ExpnData> {
        let mut prev_span = DUMMY_SP;
        iter::from_fn(move || {
            loop {
                let expn_data = self.ctxt().outer_expn_data();
                if expn_data.is_root() {
                    return None;
                }

                let is_recursive = expn_data.call_site.source_equal(prev_span);

                prev_span = self;
                self = expn_data.call_site;

                // Don't print recursive invocations.
                if !is_recursive {
                    return Some(expn_data);
                }
            }
        })
    }

    /// Splits a span into two composite spans around a certain position.
    pub fn split_at(self, pos: u32) -> (Span, Span) {
        let len = self.hi().0 - self.lo().0;
        debug_assert!(pos <= len);

        let split_pos = BytePos(self.lo().0 + pos);
        (
            Span::new(self.lo(), split_pos, self.ctxt(), self.parent()),
            Span::new(split_pos, self.hi(), self.ctxt(), self.parent()),
        )
    }

    /// Returns a `Span` that would enclose both `self` and `end`.
    ///
    /// Note that this can also be used to extend the span "backwards":
    /// `start.to(end)` and `end.to(start)` return the same `Span`.
    ///
    /// ```text
    ///     ____             ___
    ///     self lorem ipsum end
    ///     ^^^^^^^^^^^^^^^^^^^^
    /// ```
    pub fn to(self, end: Span) -> Span {
        let span_data = self.data();
        let end_data = end.data();
        // FIXME(jseyfried): `self.ctxt` should always equal `end.ctxt` here (cf. issue #23480).
        // Return the macro span on its own to avoid weird diagnostic output. It is preferable to
        // have an incomplete span than a completely nonsensical one.
        if span_data.ctxt != end_data.ctxt {
            if span_data.ctxt.is_root() {
                return end;
            } else if end_data.ctxt.is_root() {
                return self;
            }
            // Both spans fall within a macro.
            // FIXME(estebank): check if it is the *same* macro.
        }
        Span::new(
            cmp::min(span_data.lo, end_data.lo),
            cmp::max(span_data.hi, end_data.hi),
            if span_data.ctxt.is_root() { end_data.ctxt } else { span_data.ctxt },
            if span_data.parent == end_data.parent { span_data.parent } else { None },
        )
    }

    /// Returns a `Span` between the end of `self` to the beginning of `end`.
    ///
    /// ```text
    ///     ____             ___
    ///     self lorem ipsum end
    ///         ^^^^^^^^^^^^^
    /// ```
    pub fn between(self, end: Span) -> Span {
        let span = self.data();
        let end = end.data();
        Span::new(
            span.hi,
            end.lo,
            if end.ctxt.is_root() { end.ctxt } else { span.ctxt },
            if span.parent == end.parent { span.parent } else { None },
        )
    }

    /// Returns a `Span` from the beginning of `self` until the beginning of `end`.
    ///
    /// ```text
    ///     ____             ___
    ///     self lorem ipsum end
    ///     ^^^^^^^^^^^^^^^^^
    /// ```
    pub fn until(self, end: Span) -> Span {
        // Most of this function's body is copied from `to`.
        // We can't just do `self.to(end.shrink_to_lo())`,
        // because to also does some magic where it uses min/max so
        // it can handle overlapping spans. Some advanced mis-use of
        // `until` with different ctxts makes this visible.
        let span_data = self.data();
        let end_data = end.data();
        // FIXME(jseyfried): `self.ctxt` should always equal `end.ctxt` here (cf. issue #23480).
        // Return the macro span on its own to avoid weird diagnostic output. It is preferable to
        // have an incomplete span than a completely nonsensical one.
        if span_data.ctxt != end_data.ctxt {
            if span_data.ctxt.is_root() {
                return end;
            } else if end_data.ctxt.is_root() {
                return self;
            }
            // Both spans fall within a macro.
            // FIXME(estebank): check if it is the *same* macro.
        }
        Span::new(
            span_data.lo,
            end_data.lo,
            if end_data.ctxt.is_root() { end_data.ctxt } else { span_data.ctxt },
            if span_data.parent == end_data.parent { span_data.parent } else { None },
        )
    }

    pub fn from_inner(self, inner: InnerSpan) -> Span {
        let span = self.data();
        Span::new(
            span.lo + BytePos::from_usize(inner.start),
            span.lo + BytePos::from_usize(inner.end),
            span.ctxt,
            span.parent,
        )
    }

    /// Equivalent of `Span::def_site` from the proc macro API,
    /// except that the location is taken from the `self` span.
    pub fn with_def_site_ctxt(self, expn_id: ExpnId) -> Span {
        self.with_ctxt_from_mark(expn_id, Transparency::Opaque)
    }

    /// Equivalent of `Span::call_site` from the proc macro API,
    /// except that the location is taken from the `self` span.
    pub fn with_call_site_ctxt(self, expn_id: ExpnId) -> Span {
        self.with_ctxt_from_mark(expn_id, Transparency::Transparent)
    }

    /// Equivalent of `Span::mixed_site` from the proc macro API,
    /// except that the location is taken from the `self` span.
    pub fn with_mixed_site_ctxt(self, expn_id: ExpnId) -> Span {
        self.with_ctxt_from_mark(expn_id, Transparency::SemiTransparent)
    }

    /// Produces a span with the same location as `self` and context produced by a macro with the
    /// given ID and transparency, assuming that macro was defined directly and not produced by
    /// some other macro (which is the case for built-in and procedural macros).
    pub fn with_ctxt_from_mark(self, expn_id: ExpnId, transparency: Transparency) -> Span {
        self.with_ctxt(SyntaxContext::root().apply_mark(expn_id, transparency))
    }

    #[inline]
    pub fn apply_mark(self, expn_id: ExpnId, transparency: Transparency) -> Span {
        let span = self.data();
        span.with_ctxt(span.ctxt.apply_mark(expn_id, transparency))
    }

    #[inline]
    pub fn remove_mark(&mut self) -> ExpnId {
        let mut span = self.data();
        let mark = span.ctxt.remove_mark();
        *self = Span::new(span.lo, span.hi, span.ctxt, span.parent);
        mark
    }

    #[inline]
    pub fn adjust(&mut self, expn_id: ExpnId) -> Option<ExpnId> {
        let mut span = self.data();
        let mark = span.ctxt.adjust(expn_id);
        *self = Span::new(span.lo, span.hi, span.ctxt, span.parent);
        mark
    }

    #[inline]
    pub fn normalize_to_macros_2_0_and_adjust(&mut self, expn_id: ExpnId) -> Option<ExpnId> {
        let mut span = self.data();
        let mark = span.ctxt.normalize_to_macros_2_0_and_adjust(expn_id);
        *self = Span::new(span.lo, span.hi, span.ctxt, span.parent);
        mark
    }

    #[inline]
    pub fn glob_adjust(&mut self, expn_id: ExpnId, glob_span: Span) -> Option<Option<ExpnId>> {
        let mut span = self.data();
        let mark = span.ctxt.glob_adjust(expn_id, glob_span);
        *self = Span::new(span.lo, span.hi, span.ctxt, span.parent);
        mark
    }

    #[inline]
    pub fn reverse_glob_adjust(
        &mut self,
        expn_id: ExpnId,
        glob_span: Span,
    ) -> Option<Option<ExpnId>> {
        let mut span = self.data();
        let mark = span.ctxt.reverse_glob_adjust(expn_id, glob_span);
        *self = Span::new(span.lo, span.hi, span.ctxt, span.parent);
        mark
    }

    #[inline]
    pub fn normalize_to_macros_2_0(self) -> Span {
        let span = self.data();
        span.with_ctxt(span.ctxt.normalize_to_macros_2_0())
    }

    #[inline]
    pub fn normalize_to_macro_rules(self) -> Span {
        let span = self.data();
        span.with_ctxt(span.ctxt.normalize_to_macro_rules())
    }
}

impl Default for Span {
    fn default() -> Self {
        DUMMY_SP
    }
}

impl<E: Encoder> Encodable<E> for Span {
    default fn encode(&self, s: &mut E) {
        let span = self.data();
        span.lo.encode(s);
        span.hi.encode(s);
    }
}
impl<D: Decoder> Decodable<D> for Span {
    default fn decode(s: &mut D) -> Span {
        let lo = Decodable::decode(s);
        let hi = Decodable::decode(s);

        Span::new(lo, hi, SyntaxContext::root(), None)
    }
}

/// Insert `source_map` into the session globals for the duration of the
/// closure's execution.
pub fn set_source_map<T, F: FnOnce() -> T>(source_map: Lrc<SourceMap>, f: F) -> T {
    with_session_globals(|session_globals| {
        *session_globals.source_map.borrow_mut() = Some(source_map);
    });
    struct ClearSourceMap;
    impl Drop for ClearSourceMap {
        fn drop(&mut self) {
            with_session_globals(|session_globals| {
                session_globals.source_map.borrow_mut().take();
            });
        }
    }

    let _guard = ClearSourceMap;
    f()
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Use the global `SourceMap` to print the span. If that's not
        // available, fall back to printing the raw values.

        fn fallback(span: Span, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Span")
                .field("lo", &span.lo())
                .field("hi", &span.hi())
                .field("ctxt", &span.ctxt())
                .finish()
        }

        if SESSION_GLOBALS.is_set() {
            with_session_globals(|session_globals| {
                if let Some(source_map) = &*session_globals.source_map.borrow() {
                    write!(f, "{} ({:?})", source_map.span_to_diagnostic_string(*self), self.ctxt())
                } else {
                    fallback(*self, f)
                }
            })
        } else {
            fallback(*self, f)
        }
    }
}

impl fmt::Debug for SpanData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&Span::new(self.lo, self.hi, self.ctxt, self.parent), f)
    }
}

/// Identifies an offset of a multi-byte character in a `SourceFile`.
#[derive(Copy, Clone, Encodable, Decodable, Eq, PartialEq, Debug)]
pub struct MultiByteChar {
    /// The absolute offset of the character in the `SourceMap`.
    pub pos: BytePos,
    /// The number of bytes, `>= 2`.
    pub bytes: u8,
}

/// Identifies an offset of a non-narrow character in a `SourceFile`.
#[derive(Copy, Clone, Encodable, Decodable, Eq, PartialEq, Debug)]
pub enum NonNarrowChar {
    /// Represents a zero-width character.
    ZeroWidth(BytePos),
    /// Represents a wide (full-width) character.
    Wide(BytePos),
    /// Represents a tab character, represented visually with a width of 4 characters.
    Tab(BytePos),
}

impl NonNarrowChar {
    fn new(pos: BytePos, width: usize) -> Self {
        match width {
            0 => NonNarrowChar::ZeroWidth(pos),
            2 => NonNarrowChar::Wide(pos),
            4 => NonNarrowChar::Tab(pos),
            _ => panic!("width {width} given for non-narrow character"),
        }
    }

    /// Returns the absolute offset of the character in the `SourceMap`.
    pub fn pos(&self) -> BytePos {
        match *self {
            NonNarrowChar::ZeroWidth(p) | NonNarrowChar::Wide(p) | NonNarrowChar::Tab(p) => p,
        }
    }

    /// Returns the width of the character, 0 (zero-width) or 2 (wide).
    pub fn width(&self) -> usize {
        match *self {
            NonNarrowChar::ZeroWidth(_) => 0,
            NonNarrowChar::Wide(_) => 2,
            NonNarrowChar::Tab(_) => 4,
        }
    }
}

impl Add<BytePos> for NonNarrowChar {
    type Output = Self;

    fn add(self, rhs: BytePos) -> Self {
        match self {
            NonNarrowChar::ZeroWidth(pos) => NonNarrowChar::ZeroWidth(pos + rhs),
            NonNarrowChar::Wide(pos) => NonNarrowChar::Wide(pos + rhs),
            NonNarrowChar::Tab(pos) => NonNarrowChar::Tab(pos + rhs),
        }
    }
}

impl Sub<BytePos> for NonNarrowChar {
    type Output = Self;

    fn sub(self, rhs: BytePos) -> Self {
        match self {
            NonNarrowChar::ZeroWidth(pos) => NonNarrowChar::ZeroWidth(pos - rhs),
            NonNarrowChar::Wide(pos) => NonNarrowChar::Wide(pos - rhs),
            NonNarrowChar::Tab(pos) => NonNarrowChar::Tab(pos - rhs),
        }
    }
}

/// Identifies an offset of a character that was normalized away from `SourceFile`.
#[derive(Copy, Clone, Encodable, Decodable, Eq, PartialEq, Debug)]
pub struct NormalizedPos {
    /// The absolute offset of the character in the `SourceMap`.
    pub pos: BytePos,
    /// The difference between original and normalized string at position.
    pub diff: u32,
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum ExternalSource {
    /// No external source has to be loaded, since the `SourceFile` represents a local crate.
    Unneeded,
    Foreign {
        kind: ExternalSourceKind,
        /// Index of the file inside metadata.
        metadata_index: u32,
    },
}

/// The state of the lazy external source loading mechanism of a `SourceFile`.
#[derive(PartialEq, Eq, Clone, Debug)]
pub enum ExternalSourceKind {
    /// The external source has been loaded already.
    Present(Lrc<String>),
    /// No attempt has been made to load the external source.
    AbsentOk,
    /// A failed attempt has been made to load the external source.
    AbsentErr,
    Unneeded,
}

impl ExternalSource {
    pub fn get_source(&self) -> Option<&Lrc<String>> {
        match self {
            ExternalSource::Foreign { kind: ExternalSourceKind::Present(ref src), .. } => Some(src),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct OffsetOverflowError;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Encodable, Decodable)]
#[derive(HashStable_Generic)]
pub enum SourceFileHashAlgorithm {
    Md5,
    Sha1,
    Sha256,
}

impl FromStr for SourceFileHashAlgorithm {
    type Err = ();

    fn from_str(s: &str) -> Result<SourceFileHashAlgorithm, ()> {
        match s {
            "md5" => Ok(SourceFileHashAlgorithm::Md5),
            "sha1" => Ok(SourceFileHashAlgorithm::Sha1),
            "sha256" => Ok(SourceFileHashAlgorithm::Sha256),
            _ => Err(()),
        }
    }
}

/// The hash of the on-disk source file used for debug info.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[derive(HashStable_Generic, Encodable, Decodable)]
pub struct SourceFileHash {
    pub kind: SourceFileHashAlgorithm,
    value: [u8; 32],
}

impl SourceFileHash {
    pub fn new(kind: SourceFileHashAlgorithm, src: &str) -> SourceFileHash {
        let mut hash = SourceFileHash { kind, value: Default::default() };
        let len = hash.hash_len();
        let value = &mut hash.value[..len];
        let data = src.as_bytes();
        match kind {
            SourceFileHashAlgorithm::Md5 => {
                value.copy_from_slice(&Md5::digest(data));
            }
            SourceFileHashAlgorithm::Sha1 => {
                value.copy_from_slice(&Sha1::digest(data));
            }
            SourceFileHashAlgorithm::Sha256 => {
                value.copy_from_slice(&Sha256::digest(data));
            }
        }
        hash
    }

    /// Check if the stored hash matches the hash of the string.
    pub fn matches(&self, src: &str) -> bool {
        Self::new(self.kind, src) == *self
    }

    /// The bytes of the hash.
    pub fn hash_bytes(&self) -> &[u8] {
        let len = self.hash_len();
        &self.value[..len]
    }

    fn hash_len(&self) -> usize {
        match self.kind {
            SourceFileHashAlgorithm::Md5 => 16,
            SourceFileHashAlgorithm::Sha1 => 20,
            SourceFileHashAlgorithm::Sha256 => 32,
        }
    }
}

#[derive(Clone)]
pub enum SourceFileLines {
    /// The source file lines, in decoded (random-access) form.
    Lines(Vec<BytePos>),

    /// The source file lines, in undecoded difference list form.
    Diffs(SourceFileDiffs),
}

impl SourceFileLines {
    pub fn is_lines(&self) -> bool {
        matches!(self, SourceFileLines::Lines(_))
    }
}

/// The source file lines in difference list form. This matches the form
/// used within metadata, which saves space by exploiting the fact that the
/// lines list is sorted and individual lines are usually not that long.
///
/// We read it directly from metadata and only decode it into `Lines` form
/// when necessary. This is a significant performance win, especially for
/// small crates where very little of `std`'s metadata is used.
#[derive(Clone)]
pub struct SourceFileDiffs {
    /// Position of the first line. Note that this is always encoded as a
    /// `BytePos` because it is often much larger than any of the
    /// differences.
    line_start: BytePos,

    /// Always 1, 2, or 4. Always as small as possible, while being big
    /// enough to hold the length of the longest line in the source file.
    /// The 1 case is by far the most common.
    bytes_per_diff: usize,

    /// The number of diffs encoded in `raw_diffs`. Always one less than
    /// the number of lines in the source file.
    num_diffs: usize,

    /// The diffs in "raw" form. Each segment of `bytes_per_diff` length
    /// encodes one little-endian diff. Note that they aren't LEB128
    /// encoded. This makes for much faster decoding. Besides, the
    /// bytes_per_diff==1 case is by far the most common, and LEB128
    /// encoding has no effect on that case.
    raw_diffs: Vec<u8>,
}

/// A single source in the [`SourceMap`].
pub struct SourceFile {
    /// The name of the file that the source came from. Source that doesn't
    /// originate from files has names between angle brackets by convention
    /// (e.g., `<anon>`).
    pub name: FileName,
    /// The complete source code.
    pub src: Option<Lrc<String>>,
    /// The source code's hash.
    pub src_hash: SourceFileHash,
    /// The external source code (used for external crates, which will have a `None`
    /// value as `self.src`.
    pub external_src: Lock<ExternalSource>,
    /// The start position of this source in the `SourceMap`.
    pub start_pos: BytePos,
    /// The end position of this source in the `SourceMap`.
    pub end_pos: BytePos,
    /// Locations of lines beginnings in the source code.
    pub lines: Lock<SourceFileLines>,
    /// Locations of multi-byte characters in the source code.
    pub multibyte_chars: Vec<MultiByteChar>,
    /// Width of characters that are not narrow in the source code.
    pub non_narrow_chars: Vec<NonNarrowChar>,
    /// Locations of characters removed during normalization.
    pub normalized_pos: Vec<NormalizedPos>,
    /// A hash of the filename, used for speeding up hashing in incremental compilation.
    pub name_hash: Hash128,
    /// Indicates which crate this `SourceFile` was imported from.
    pub cnum: CrateNum,
}

impl Clone for SourceFile {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            src: self.src.clone(),
            src_hash: self.src_hash,
            external_src: Lock::new(self.external_src.borrow().clone()),
            start_pos: self.start_pos,
            end_pos: self.end_pos,
            lines: Lock::new(self.lines.borrow().clone()),
            multibyte_chars: self.multibyte_chars.clone(),
            non_narrow_chars: self.non_narrow_chars.clone(),
            normalized_pos: self.normalized_pos.clone(),
            name_hash: self.name_hash,
            cnum: self.cnum,
        }
    }
}

impl<S: Encoder> Encodable<S> for SourceFile {
    fn encode(&self, s: &mut S) {
        self.name.encode(s);
        self.src_hash.encode(s);
        self.start_pos.encode(s);
        self.end_pos.encode(s);

        // We are always in `Lines` form by the time we reach here.
        assert!(self.lines.borrow().is_lines());
        self.lines(|lines| {
            // Store the length.
            s.emit_u32(lines.len() as u32);

            // Compute and store the difference list.
            if lines.len() != 0 {
                let max_line_length = if lines.len() == 1 {
                    0
                } else {
                    lines
                        .array_windows()
                        .map(|&[fst, snd]| snd - fst)
                        .map(|bp| bp.to_usize())
                        .max()
                        .unwrap()
                };

                let bytes_per_diff: usize = match max_line_length {
                    0..=0xFF => 1,
                    0x100..=0xFFFF => 2,
                    _ => 4,
                };

                // Encode the number of bytes used per diff.
                s.emit_u8(bytes_per_diff as u8);

                // Encode the first element.
                lines[0].encode(s);

                // Encode the difference list.
                let diff_iter = lines.array_windows().map(|&[fst, snd]| snd - fst);
                let num_diffs = lines.len() - 1;
                let mut raw_diffs;
                match bytes_per_diff {
                    1 => {
                        raw_diffs = Vec::with_capacity(num_diffs);
                        for diff in diff_iter {
                            raw_diffs.push(diff.0 as u8);
                        }
                    }
                    2 => {
                        raw_diffs = Vec::with_capacity(bytes_per_diff * num_diffs);
                        for diff in diff_iter {
                            raw_diffs.extend_from_slice(&(diff.0 as u16).to_le_bytes());
                        }
                    }
                    4 => {
                        raw_diffs = Vec::with_capacity(bytes_per_diff * num_diffs);
                        for diff in diff_iter {
                            raw_diffs.extend_from_slice(&(diff.0).to_le_bytes());
                        }
                    }
                    _ => unreachable!(),
                }
                s.emit_raw_bytes(&raw_diffs);
            }
        });

        self.multibyte_chars.encode(s);
        self.non_narrow_chars.encode(s);
        self.name_hash.encode(s);
        self.normalized_pos.encode(s);
        self.cnum.encode(s);
    }
}

impl<D: Decoder> Decodable<D> for SourceFile {
    fn decode(d: &mut D) -> SourceFile {
        let name: FileName = Decodable::decode(d);
        let src_hash: SourceFileHash = Decodable::decode(d);
        let start_pos: BytePos = Decodable::decode(d);
        let end_pos: BytePos = Decodable::decode(d);
        let lines = {
            let num_lines: u32 = Decodable::decode(d);
            if num_lines > 0 {
                // Read the number of bytes used per diff.
                let bytes_per_diff = d.read_u8() as usize;

                // Read the first element.
                let line_start: BytePos = Decodable::decode(d);

                // Read the difference list.
                let num_diffs = num_lines as usize - 1;
                let raw_diffs = d.read_raw_bytes(bytes_per_diff * num_diffs).to_vec();
                SourceFileLines::Diffs(SourceFileDiffs {
                    line_start,
                    bytes_per_diff,
                    num_diffs,
                    raw_diffs,
                })
            } else {
                SourceFileLines::Lines(vec![])
            }
        };
        let multibyte_chars: Vec<MultiByteChar> = Decodable::decode(d);
        let non_narrow_chars: Vec<NonNarrowChar> = Decodable::decode(d);
        let name_hash = Decodable::decode(d);
        let normalized_pos: Vec<NormalizedPos> = Decodable::decode(d);
        let cnum: CrateNum = Decodable::decode(d);
        SourceFile {
            name,
            start_pos,
            end_pos,
            src: None,
            src_hash,
            // Unused - the metadata decoder will construct
            // a new SourceFile, filling in `external_src` properly
            external_src: Lock::new(ExternalSource::Unneeded),
            lines: Lock::new(lines),
            multibyte_chars,
            non_narrow_chars,
            normalized_pos,
            name_hash,
            cnum,
        }
    }
}

impl fmt::Debug for SourceFile {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "SourceFile({:?})", self.name)
    }
}

impl SourceFile {
    pub fn new(
        name: FileName,
        mut src: String,
        start_pos: BytePos,
        hash_kind: SourceFileHashAlgorithm,
    ) -> Self {
        // Compute the file hash before any normalization.
        let src_hash = SourceFileHash::new(hash_kind, &src);
        let normalized_pos = normalize_src(&mut src, start_pos);

        let name_hash = {
            let mut hasher: StableHasher = StableHasher::new();
            name.hash(&mut hasher);
            hasher.finish()
        };
        let end_pos = start_pos.to_usize() + src.len();
        assert!(end_pos <= u32::MAX as usize);

        let (lines, multibyte_chars, non_narrow_chars) =
            analyze_source_file::analyze_source_file(&src, start_pos);

        SourceFile {
            name,
            src: Some(Lrc::new(src)),
            src_hash,
            external_src: Lock::new(ExternalSource::Unneeded),
            start_pos,
            end_pos: Pos::from_usize(end_pos),
            lines: Lock::new(SourceFileLines::Lines(lines)),
            multibyte_chars,
            non_narrow_chars,
            normalized_pos,
            name_hash,
            cnum: LOCAL_CRATE,
        }
    }

    pub fn lines<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&[BytePos]) -> R,
    {
        let mut guard = self.lines.borrow_mut();
        match &*guard {
            SourceFileLines::Lines(lines) => f(lines),
            SourceFileLines::Diffs(SourceFileDiffs {
                mut line_start,
                bytes_per_diff,
                num_diffs,
                raw_diffs,
            }) => {
                // Convert from "diffs" form to "lines" form.
                let num_lines = num_diffs + 1;
                let mut lines = Vec::with_capacity(num_lines);
                lines.push(line_start);

                assert_eq!(*num_diffs, raw_diffs.len() / bytes_per_diff);
                match bytes_per_diff {
                    1 => {
                        lines.extend(raw_diffs.into_iter().map(|&diff| {
                            line_start = line_start + BytePos(diff as u32);
                            line_start
                        }));
                    }
                    2 => {
                        lines.extend((0..*num_diffs).map(|i| {
                            let pos = bytes_per_diff * i;
                            let bytes = [raw_diffs[pos], raw_diffs[pos + 1]];
                            let diff = u16::from_le_bytes(bytes);
                            line_start = line_start + BytePos(diff as u32);
                            line_start
                        }));
                    }
                    4 => {
                        lines.extend((0..*num_diffs).map(|i| {
                            let pos = bytes_per_diff * i;
                            let bytes = [
                                raw_diffs[pos],
                                raw_diffs[pos + 1],
                                raw_diffs[pos + 2],
                                raw_diffs[pos + 3],
                            ];
                            let diff = u32::from_le_bytes(bytes);
                            line_start = line_start + BytePos(diff);
                            line_start
                        }));
                    }
                    _ => unreachable!(),
                }
                let res = f(&lines);
                *guard = SourceFileLines::Lines(lines);
                res
            }
        }
    }

    /// Returns the `BytePos` of the beginning of the current line.
    pub fn line_begin_pos(&self, pos: BytePos) -> BytePos {
        let line_index = self.lookup_line(pos).unwrap();
        self.lines(|lines| lines[line_index])
    }

    /// Add externally loaded source.
    /// If the hash of the input doesn't match or no input is supplied via None,
    /// it is interpreted as an error and the corresponding enum variant is set.
    /// The return value signifies whether some kind of source is present.
    pub fn add_external_src<F>(&self, get_src: F) -> bool
    where
        F: FnOnce() -> Option<String>,
    {
        if matches!(
            *self.external_src.borrow(),
            ExternalSource::Foreign { kind: ExternalSourceKind::AbsentOk, .. }
        ) {
            let src = get_src();
            let mut external_src = self.external_src.borrow_mut();
            // Check that no-one else have provided the source while we were getting it
            if let ExternalSource::Foreign {
                kind: src_kind @ ExternalSourceKind::AbsentOk, ..
            } = &mut *external_src
            {
                if let Some(mut src) = src {
                    // The src_hash needs to be computed on the pre-normalized src.
                    if self.src_hash.matches(&src) {
                        normalize_src(&mut src, BytePos::from_usize(0));
                        *src_kind = ExternalSourceKind::Present(Lrc::new(src));
                        return true;
                    }
                } else {
                    *src_kind = ExternalSourceKind::AbsentErr;
                }

                false
            } else {
                self.src.is_some() || external_src.get_source().is_some()
            }
        } else {
            self.src.is_some() || self.external_src.borrow().get_source().is_some()
        }
    }

    /// Gets a line from the list of pre-computed line-beginnings.
    /// The line number here is 0-based.
    pub fn get_line(&self, line_number: usize) -> Option<Cow<'_, str>> {
        fn get_until_newline(src: &str, begin: usize) -> &str {
            // We can't use `lines.get(line_number+1)` because we might
            // be parsing when we call this function and thus the current
            // line is the last one we have line info for.
            let slice = &src[begin..];
            match slice.find('\n') {
                Some(e) => &slice[..e],
                None => slice,
            }
        }

        let begin = {
            let line = self.lines(|lines| lines.get(line_number).copied())?;
            let begin: BytePos = line - self.start_pos;
            begin.to_usize()
        };

        if let Some(ref src) = self.src {
            Some(Cow::from(get_until_newline(src, begin)))
        } else {
            self.external_src
                .borrow()
                .get_source()
                .map(|src| Cow::Owned(String::from(get_until_newline(src, begin))))
        }
    }

    pub fn is_real_file(&self) -> bool {
        self.name.is_real()
    }

    #[inline]
    pub fn is_imported(&self) -> bool {
        self.src.is_none()
    }

    pub fn count_lines(&self) -> usize {
        self.lines(|lines| lines.len())
    }

    /// Finds the line containing the given position. The return value is the
    /// index into the `lines` array of this `SourceFile`, not the 1-based line
    /// number. If the source_file is empty or the position is located before the
    /// first line, `None` is returned.
    pub fn lookup_line(&self, pos: BytePos) -> Option<usize> {
        self.lines(|lines| lines.partition_point(|x| x <= &pos).checked_sub(1))
    }

    pub fn line_bounds(&self, line_index: usize) -> Range<BytePos> {
        if self.is_empty() {
            return self.start_pos..self.end_pos;
        }

        self.lines(|lines| {
            assert!(line_index < lines.len());
            if line_index == (lines.len() - 1) {
                lines[line_index]..self.end_pos
            } else {
                lines[line_index]..lines[line_index + 1]
            }
        })
    }

    /// Returns whether or not the file contains the given `SourceMap` byte
    /// position. The position one past the end of the file is considered to be
    /// contained by the file. This implies that files for which `is_empty`
    /// returns true still contain one byte position according to this function.
    #[inline]
    pub fn contains(&self, byte_pos: BytePos) -> bool {
        byte_pos >= self.start_pos && byte_pos <= self.end_pos
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start_pos == self.end_pos
    }

    /// Calculates the original byte position relative to the start of the file
    /// based on the given byte position.
    pub fn original_relative_byte_pos(&self, pos: BytePos) -> BytePos {
        // Diff before any records is 0. Otherwise use the previously recorded
        // diff as that applies to the following characters until a new diff
        // is recorded.
        let diff = match self.normalized_pos.binary_search_by(|np| np.pos.cmp(&pos)) {
            Ok(i) => self.normalized_pos[i].diff,
            Err(i) if i == 0 => 0,
            Err(i) => self.normalized_pos[i - 1].diff,
        };

        BytePos::from_u32(pos.0 - self.start_pos.0 + diff)
    }

    /// Calculates a normalized byte position from a byte offset relative to the
    /// start of the file.
    ///
    /// When we get an inline assembler error from LLVM during codegen, we
    /// import the expanded assembly code as a new `SourceFile`, which can then
    /// be used for error reporting with spans. However the byte offsets given
    /// to us by LLVM are relative to the start of the original buffer, not the
    /// normalized one. Hence we need to convert those offsets to the normalized
    /// form when constructing spans.
    pub fn normalized_byte_pos(&self, offset: u32) -> BytePos {
        let diff = match self
            .normalized_pos
            .binary_search_by(|np| (np.pos.0 + np.diff).cmp(&(self.start_pos.0 + offset)))
        {
            Ok(i) => self.normalized_pos[i].diff,
            Err(i) if i == 0 => 0,
            Err(i) => self.normalized_pos[i - 1].diff,
        };

        BytePos::from_u32(self.start_pos.0 + offset - diff)
    }

    /// Converts an absolute `BytePos` to a `CharPos` relative to the `SourceFile`.
    pub fn bytepos_to_file_charpos(&self, bpos: BytePos) -> CharPos {
        // The number of extra bytes due to multibyte chars in the `SourceFile`.
        let mut total_extra_bytes = 0;

        for mbc in self.multibyte_chars.iter() {
            debug!("{}-byte char at {:?}", mbc.bytes, mbc.pos);
            if mbc.pos < bpos {
                // Every character is at least one byte, so we only
                // count the actual extra bytes.
                total_extra_bytes += mbc.bytes as u32 - 1;
                // We should never see a byte position in the middle of a
                // character.
                assert!(bpos.to_u32() >= mbc.pos.to_u32() + mbc.bytes as u32);
            } else {
                break;
            }
        }

        assert!(self.start_pos.to_u32() + total_extra_bytes <= bpos.to_u32());
        CharPos(bpos.to_usize() - self.start_pos.to_usize() - total_extra_bytes as usize)
    }

    /// Looks up the file's (1-based) line number and (0-based `CharPos`) column offset, for a
    /// given `BytePos`.
    pub fn lookup_file_pos(&self, pos: BytePos) -> (usize, CharPos) {
        let chpos = self.bytepos_to_file_charpos(pos);
        match self.lookup_line(pos) {
            Some(a) => {
                let line = a + 1; // Line numbers start at 1
                let linebpos = self.lines(|lines| lines[a]);
                let linechpos = self.bytepos_to_file_charpos(linebpos);
                let col = chpos - linechpos;
                debug!("byte pos {:?} is on the line at byte pos {:?}", pos, linebpos);
                debug!("char pos {:?} is on the line at char pos {:?}", chpos, linechpos);
                debug!("byte is on line: {}", line);
                assert!(chpos >= linechpos);
                (line, col)
            }
            None => (0, chpos),
        }
    }

    /// Looks up the file's (1-based) line number, (0-based `CharPos`) column offset, and (0-based)
    /// column offset when displayed, for a given `BytePos`.
    pub fn lookup_file_pos_with_col_display(&self, pos: BytePos) -> (usize, CharPos, usize) {
        let (line, col_or_chpos) = self.lookup_file_pos(pos);
        if line > 0 {
            let col = col_or_chpos;
            let linebpos = self.lines(|lines| lines[line - 1]);
            let col_display = {
                let start_width_idx = self
                    .non_narrow_chars
                    .binary_search_by_key(&linebpos, |x| x.pos())
                    .unwrap_or_else(|x| x);
                let end_width_idx = self
                    .non_narrow_chars
                    .binary_search_by_key(&pos, |x| x.pos())
                    .unwrap_or_else(|x| x);
                let special_chars = end_width_idx - start_width_idx;
                let non_narrow: usize = self.non_narrow_chars[start_width_idx..end_width_idx]
                    .iter()
                    .map(|x| x.width())
                    .sum();
                col.0 - special_chars + non_narrow
            };
            (line, col, col_display)
        } else {
            let chpos = col_or_chpos;
            let col_display = {
                let end_width_idx = self
                    .non_narrow_chars
                    .binary_search_by_key(&pos, |x| x.pos())
                    .unwrap_or_else(|x| x);
                let non_narrow: usize =
                    self.non_narrow_chars[0..end_width_idx].iter().map(|x| x.width()).sum();
                chpos.0 - end_width_idx + non_narrow
            };
            (0, chpos, col_display)
        }
    }
}

/// Normalizes the source code and records the normalizations.
fn normalize_src(src: &mut String, start_pos: BytePos) -> Vec<NormalizedPos> {
    let mut normalized_pos = vec![];
    remove_bom(src, &mut normalized_pos);
    normalize_newlines(src, &mut normalized_pos);

    // Offset all the positions by start_pos to match the final file positions.
    for np in &mut normalized_pos {
        np.pos.0 += start_pos.0;
    }

    normalized_pos
}

/// Removes UTF-8 BOM, if any.
fn remove_bom(src: &mut String, normalized_pos: &mut Vec<NormalizedPos>) {
    if src.starts_with('\u{feff}') {
        src.drain(..3);
        normalized_pos.push(NormalizedPos { pos: BytePos(0), diff: 3 });
    }
}

/// Replaces `\r\n` with `\n` in-place in `src`.
///
/// Returns error if there's a lone `\r` in the string.
fn normalize_newlines(src: &mut String, normalized_pos: &mut Vec<NormalizedPos>) {
    if !src.as_bytes().contains(&b'\r') {
        return;
    }

    // We replace `\r\n` with `\n` in-place, which doesn't break utf-8 encoding.
    // While we *can* call `as_mut_vec` and do surgery on the live string
    // directly, let's rather steal the contents of `src`. This makes the code
    // safe even if a panic occurs.

    let mut buf = std::mem::replace(src, String::new()).into_bytes();
    let mut gap_len = 0;
    let mut tail = buf.as_mut_slice();
    let mut cursor = 0;
    let original_gap = normalized_pos.last().map_or(0, |l| l.diff);
    loop {
        let idx = match find_crlf(&tail[gap_len..]) {
            None => tail.len(),
            Some(idx) => idx + gap_len,
        };
        tail.copy_within(gap_len..idx, 0);
        tail = &mut tail[idx - gap_len..];
        if tail.len() == gap_len {
            break;
        }
        cursor += idx - gap_len;
        gap_len += 1;
        normalized_pos.push(NormalizedPos {
            pos: BytePos::from_usize(cursor + 1),
            diff: original_gap + gap_len as u32,
        });
    }

    // Account for removed `\r`.
    // After `set_len`, `buf` is guaranteed to contain utf-8 again.
    let new_len = buf.len() - gap_len;
    unsafe {
        buf.set_len(new_len);
        *src = String::from_utf8_unchecked(buf);
    }

    fn find_crlf(src: &[u8]) -> Option<usize> {
        let mut search_idx = 0;
        while let Some(idx) = find_cr(&src[search_idx..]) {
            if src[search_idx..].get(idx + 1) != Some(&b'\n') {
                search_idx += idx + 1;
                continue;
            }
            return Some(search_idx + idx);
        }
        None
    }

    fn find_cr(src: &[u8]) -> Option<usize> {
        src.iter().position(|&b| b == b'\r')
    }
}

// _____________________________________________________________________________
// Pos, BytePos, CharPos
//

pub trait Pos {
    fn from_usize(n: usize) -> Self;
    fn to_usize(&self) -> usize;
    fn from_u32(n: u32) -> Self;
    fn to_u32(&self) -> u32;
}

macro_rules! impl_pos {
    (
        $(
            $(#[$attr:meta])*
            $vis:vis struct $ident:ident($inner_vis:vis $inner_ty:ty);
        )*
    ) => {
        $(
            $(#[$attr])*
            $vis struct $ident($inner_vis $inner_ty);

            impl Pos for $ident {
                #[inline(always)]
                fn from_usize(n: usize) -> $ident {
                    $ident(n as $inner_ty)
                }

                #[inline(always)]
                fn to_usize(&self) -> usize {
                    self.0 as usize
                }

                #[inline(always)]
                fn from_u32(n: u32) -> $ident {
                    $ident(n as $inner_ty)
                }

                #[inline(always)]
                fn to_u32(&self) -> u32 {
                    self.0 as u32
                }
            }

            impl Add for $ident {
                type Output = $ident;

                #[inline(always)]
                fn add(self, rhs: $ident) -> $ident {
                    $ident(self.0 + rhs.0)
                }
            }

            impl Sub for $ident {
                type Output = $ident;

                #[inline(always)]
                fn sub(self, rhs: $ident) -> $ident {
                    $ident(self.0 - rhs.0)
                }
            }
        )*
    };
}

impl_pos! {
    /// A byte offset.
    ///
    /// Keep this small (currently 32-bits), as AST contains a lot of them.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
    pub struct BytePos(pub u32);

    /// A character offset.
    ///
    /// Because of multibyte UTF-8 characters, a byte offset
    /// is not equivalent to a character offset. The [`SourceMap`] will convert [`BytePos`]
    /// values to `CharPos` values as necessary.
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
    pub struct CharPos(pub usize);
}

impl<S: Encoder> Encodable<S> for BytePos {
    fn encode(&self, s: &mut S) {
        s.emit_u32(self.0);
    }
}

impl<D: Decoder> Decodable<D> for BytePos {
    fn decode(d: &mut D) -> BytePos {
        BytePos(d.read_u32())
    }
}

// _____________________________________________________________________________
// Loc, SourceFileAndLine, SourceFileAndBytePos
//

/// A source code location used for error reporting.
#[derive(Debug, Clone)]
pub struct Loc {
    /// Information about the original source.
    pub file: Lrc<SourceFile>,
    /// The (1-based) line number.
    pub line: usize,
    /// The (0-based) column offset.
    pub col: CharPos,
    /// The (0-based) column offset when displayed.
    pub col_display: usize,
}

// Used to be structural records.
#[derive(Debug)]
pub struct SourceFileAndLine {
    pub sf: Lrc<SourceFile>,
    /// Index of line, starting from 0.
    pub line: usize,
}
#[derive(Debug)]
pub struct SourceFileAndBytePos {
    pub sf: Lrc<SourceFile>,
    pub pos: BytePos,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LineInfo {
    /// Index of line, starting from 0.
    pub line_index: usize,

    /// Column in line where span begins, starting from 0.
    pub start_col: CharPos,

    /// Column in line where span ends, starting from 0, exclusive.
    pub end_col: CharPos,
}

pub struct FileLines {
    pub file: Lrc<SourceFile>,
    pub lines: Vec<LineInfo>,
}

pub static SPAN_TRACK: AtomicRef<fn(LocalDefId)> = AtomicRef::new(&((|_| {}) as fn(_)));

// _____________________________________________________________________________
// SpanLinesError, SpanSnippetError, DistinctSources, MalformedSourceMapPositions
//

pub type FileLinesResult = Result<FileLines, SpanLinesError>;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SpanLinesError {
    DistinctSources(Box<DistinctSources>),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SpanSnippetError {
    IllFormedSpan(Span),
    DistinctSources(Box<DistinctSources>),
    MalformedForSourcemap(MalformedSourceMapPositions),
    SourceNotAvailable { filename: FileName },
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DistinctSources {
    pub begin: (FileName, BytePos),
    pub end: (FileName, BytePos),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MalformedSourceMapPositions {
    pub name: FileName,
    pub source_len: usize,
    pub begin_pos: BytePos,
    pub end_pos: BytePos,
}

/// Range inside of a `Span` used for diagnostics when we only have access to relative positions.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct InnerSpan {
    pub start: usize,
    pub end: usize,
}

impl InnerSpan {
    pub fn new(start: usize, end: usize) -> InnerSpan {
        InnerSpan { start, end }
    }
}

/// Requirements for a `StableHashingContext` to be used in this crate.
///
/// This is a hack to allow using the [`HashStable_Generic`] derive macro
/// instead of implementing everything in rustc_middle.
pub trait HashStableContext {
    fn def_path_hash(&self, def_id: DefId) -> DefPathHash;
    fn hash_spans(&self) -> bool;
    /// Accesses `sess.opts.unstable_opts.incremental_ignore_spans` since
    /// we don't have easy access to a `Session`
    fn unstable_opts_incremental_ignore_spans(&self) -> bool;
    fn def_span(&self, def_id: LocalDefId) -> Span;
    fn span_data_to_lines_and_cols(
        &mut self,
        span: &SpanData,
    ) -> Option<(Lrc<SourceFile>, usize, BytePos, usize, BytePos)>;
    fn hashing_controls(&self) -> HashingControls;
}

impl<CTX> HashStable<CTX> for Span
where
    CTX: HashStableContext,
{
    /// Hashes a span in a stable way. We can't directly hash the span's `BytePos`
    /// fields (that would be similar to hashing pointers, since those are just
    /// offsets into the `SourceMap`). Instead, we hash the (file name, line, column)
    /// triple, which stays the same even if the containing `SourceFile` has moved
    /// within the `SourceMap`.
    ///
    /// Also note that we are hashing byte offsets for the column, not unicode
    /// codepoint offsets. For the purpose of the hash that's sufficient.
    /// Also, hashing filenames is expensive so we avoid doing it twice when the
    /// span starts and ends in the same file, which is almost always the case.
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        const TAG_VALID_SPAN: u8 = 0;
        const TAG_INVALID_SPAN: u8 = 1;
        const TAG_RELATIVE_SPAN: u8 = 2;

        if !ctx.hash_spans() {
            return;
        }

        let span = self.data_untracked();
        span.ctxt.hash_stable(ctx, hasher);
        span.parent.hash_stable(ctx, hasher);

        if span.is_dummy() {
            Hash::hash(&TAG_INVALID_SPAN, hasher);
            return;
        }

        if let Some(parent) = span.parent {
            let def_span = ctx.def_span(parent).data_untracked();
            if def_span.contains(span) {
                // This span is enclosed in a definition: only hash the relative position.
                Hash::hash(&TAG_RELATIVE_SPAN, hasher);
                (span.lo - def_span.lo).to_u32().hash_stable(ctx, hasher);
                (span.hi - def_span.lo).to_u32().hash_stable(ctx, hasher);
                return;
            }
        }

        // If this is not an empty or invalid span, we want to hash the last
        // position that belongs to it, as opposed to hashing the first
        // position past it.
        let Some((file, line_lo, col_lo, line_hi, col_hi)) = ctx.span_data_to_lines_and_cols(&span)
        else {
            Hash::hash(&TAG_INVALID_SPAN, hasher);
            return;
        };

        Hash::hash(&TAG_VALID_SPAN, hasher);
        Hash::hash(&file.name_hash, hasher);

        // Hash both the length and the end location (line/column) of a span. If we
        // hash only the length, for example, then two otherwise equal spans with
        // different end locations will have the same hash. This can cause a problem
        // during incremental compilation wherein a previous result for a query that
        // depends on the end location of a span will be incorrectly reused when the
        // end location of the span it depends on has changed (see issue #74890). A
        // similar analysis applies if some query depends specifically on the length
        // of the span, but we only hash the end location. So hash both.

        let col_lo_trunc = (col_lo.0 as u64) & 0xFF;
        let line_lo_trunc = ((line_lo as u64) & 0xFF_FF_FF) << 8;
        let col_hi_trunc = (col_hi.0 as u64) & 0xFF << 32;
        let line_hi_trunc = ((line_hi as u64) & 0xFF_FF_FF) << 40;
        let col_line = col_lo_trunc | line_lo_trunc | col_hi_trunc | line_hi_trunc;
        let len = (span.hi - span.lo).0;
        Hash::hash(&col_line, hasher);
        Hash::hash(&len, hasher);
    }
}

/// Useful type to use with `Result<>` indicate that an error has already
/// been reported to the user, so no need to continue checking.
#[derive(Clone, Copy, Debug, Encodable, Decodable, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[derive(HashStable_Generic)]
pub struct ErrorGuaranteed(());

impl ErrorGuaranteed {
    /// To be used only if you really know what you are doing... ideally, we would find a way to
    /// eliminate all calls to this method.
    #[deprecated = "`Session::delay_span_bug` should be preferred over this function"]
    pub fn unchecked_claim_error_was_emitted() -> Self {
        ErrorGuaranteed(())
    }
}
