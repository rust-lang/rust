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
#![feature(crate_visibility_modifier)]
#![feature(if_let_guard)]
#![feature(negative_impls)]
#![feature(nll)]
#![feature(min_specialization)]
#![feature(thread_local_const_init)]

#[macro_use]
extern crate rustc_macros;

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
pub use hygiene::{DesugaringKind, ExpnKind, ForLoopLoc, MacroKind};
pub use hygiene::{ExpnData, ExpnHash, ExpnId, LocalExpnId, SyntaxContext};
pub mod def_id;
use def_id::{CrateNum, DefId, DefPathHash, LOCAL_CRATE};
pub mod lev_distance;
mod span_encoding;
pub use span_encoding::{Span, DUMMY_SP};

pub mod symbol;
pub use symbol::{sym, Symbol};

mod analyze_source_file;
pub mod fatal_error;

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::{Lock, Lrc};

use std::borrow::Cow;
use std::cmp::{self, Ordering};
use std::fmt;
use std::hash::Hash;
use std::ops::{Add, Range, Sub};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use md5::Md5;
use sha1::Digest;
use sha1::Sha1;
use sha2::Sha256;

use tracing::debug;

#[cfg(test)]
mod tests;

// Per-session global variables: this struct is stored in thread-local storage
// in such a way that it is accessible without any kind of handle to all
// threads within the compilation session, but is not accessible outside the
// session.
pub struct SessionGlobals {
    symbol_interner: Lock<symbol::Interner>,
    span_interner: Lock<span_encoding::SpanInterner>,
    hygiene_data: Lock<hygiene::HygieneData>,
    source_map: Lock<Option<Lrc<SourceMap>>>,
}

impl SessionGlobals {
    pub fn new(edition: Edition) -> SessionGlobals {
        SessionGlobals {
            symbol_interner: Lock::new(symbol::Interner::fresh()),
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
    fn encode(&self, encoder: &mut S) -> Result<(), S::Error> {
        encoder.emit_enum(|encoder| match *self {
            RealFileName::LocalPath(ref local_path) => {
                encoder.emit_enum_variant("LocalPath", 0, 1, |encoder| {
                    Ok({
                        encoder
                            .emit_enum_variant_arg(true, |encoder| local_path.encode(encoder))?;
                    })
                })
            }

            RealFileName::Remapped { ref local_path, ref virtual_name } => encoder
                .emit_enum_variant("Remapped", 1, 2, |encoder| {
                    // For privacy and build reproducibility, we must not embed host-dependant path in artifacts
                    // if they have been remapped by --remap-path-prefix
                    assert!(local_path.is_none());
                    Ok({
                        encoder
                            .emit_enum_variant_arg(true, |encoder| local_path.encode(encoder))?;
                        encoder
                            .emit_enum_variant_arg(false, |encoder| virtual_name.encode(encoder))?;
                    })
                }),
        })
    }
}

impl RealFileName {
    /// Returns the path suitable for reading from the file system on the local host,
    /// if this information exists.
    /// Avoid embedding this in build artifacts; see `remapped_path_if_available()` for that.
    pub fn local_path(&self) -> Option<&Path> {
        match self {
            RealFileName::LocalPath(p) => Some(p),
            RealFileName::Remapped { local_path: p, virtual_name: _ } => {
                p.as_ref().map(PathBuf::as_path)
            }
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
            | RealFileName::Remapped { local_path: _, virtual_name: p } => &p,
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
        }
    }
}

/// Differentiates between real files and common virtual files.
#[derive(Debug, Eq, PartialEq, Clone, Ord, PartialOrd, Hash)]
#[derive(Decodable, Encodable)]
pub enum FileName {
    Real(RealFileName),
    /// Call to `quote!`.
    QuoteExpansion(u64),
    /// Command line.
    Anon(u64),
    /// Hack in `src/librustc_ast/parse.rs`.
    // FIXME(jseyfried)
    MacroExpansion(u64),
    ProcMacroSourceCode(u64),
    /// Strings provided as `--cfg [cfgspec]` stored in a `crate_cfg`.
    CfgSpec(u64),
    /// Strings provided as crate attributes in the CLI.
    CliCrateAttr(u64),
    /// Custom sources for explicit parser calls from plugins and drivers.
    Custom(String),
    DocTest(PathBuf, isize),
    /// Post-substitution inline assembly from LLVM.
    InlineAsm(u64),
}

impl From<PathBuf> for FileName {
    fn from(p: PathBuf) -> Self {
        assert!(!p.to_string_lossy().ends_with('>'));
        FileName::Real(RealFileName::LocalPath(p))
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub enum FileNameDisplayPreference {
    Remapped,
    Local,
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
            Custom(ref s) => write!(fmt, "<{}>", s),
            DocTest(ref path, _) => write!(fmt, "{}", path.display()),
            InlineAsm(_) => write!(fmt, "<inline asm>"),
        }
    }
}

impl FileNameDisplay<'_> {
    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        match self.inner {
            FileName::Real(ref inner) => inner.to_string_lossy(self.display_pref),
            _ => Cow::from(format!("{}", self)),
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

    // This may include transient local filesystem information.
    // Must not be embedded in build outputs.
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
#[derive(Clone, Copy, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub struct SpanData {
    pub lo: BytePos,
    pub hi: BytePos,
    /// Information about where the macro came from, if this piece of
    /// code was created by a macro expansion.
    pub ctxt: SyntaxContext,
}

impl SpanData {
    #[inline]
    pub fn span(&self) -> Span {
        Span::new(self.lo, self.hi, self.ctxt)
    }
    #[inline]
    pub fn with_lo(&self, lo: BytePos) -> Span {
        Span::new(lo, self.hi, self.ctxt)
    }
    #[inline]
    pub fn with_hi(&self, hi: BytePos) -> Span {
        Span::new(self.lo, hi, self.ctxt)
    }
    #[inline]
    pub fn with_ctxt(&self, ctxt: SyntaxContext) -> Span {
        Span::new(self.lo, self.hi, ctxt)
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

/// A collection of `Span`s.
///
/// Spans have two orthogonal attributes:
///
/// - They can be *primary spans*. In this case they are the locus of
///   the error, and would be rendered with `^^^`.
/// - They can have a *label*. In this case, the label is written next
///   to the mark in the snippet when we render.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Encodable, Decodable)]
pub struct MultiSpan {
    primary_spans: Vec<Span>,
    span_labels: Vec<(Span, String)>,
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
    pub fn ctxt(self) -> SyntaxContext {
        self.data().ctxt
    }
    #[inline]
    pub fn with_ctxt(self, ctxt: SyntaxContext) -> Span {
        self.data().with_ctxt(ctxt)
    }

    /// Returns `true` if this is a dummy span with any hygienic context.
    #[inline]
    pub fn is_dummy(self) -> bool {
        let span = self.data();
        span.lo.0 == 0 && span.hi.0 == 0
    }

    /// Returns `true` if this span comes from a macro or desugaring.
    #[inline]
    pub fn from_expansion(self) -> bool {
        self.ctxt() != SyntaxContext::root()
    }

    /// Returns `true` if `span` originates in a derive-macro's expansion.
    pub fn in_derive_expansion(self) -> bool {
        matches!(self.ctxt().outer_expn_data().kind, ExpnKind::Macro(MacroKind::Derive, _))
    }

    #[inline]
    pub fn with_root_ctxt(lo: BytePos, hi: BytePos) -> Span {
        Span::new(lo, hi, SyntaxContext::root())
    }

    /// Returns a new span representing an empty span at the beginning of this span.
    #[inline]
    pub fn shrink_to_lo(self) -> Span {
        let span = self.data();
        span.with_hi(span.lo)
    }
    /// Returns a new span representing an empty span at the end of this span.
    #[inline]
    pub fn shrink_to_hi(self) -> Span {
        let span = self.data();
        span.with_lo(span.hi)
    }

    #[inline]
    /// Returns `true` if `hi == lo`.
    pub fn is_empty(&self) -> bool {
        let span = self.data();
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
        span.lo <= other.lo && other.hi <= span.hi
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
    pub fn source_equal(&self, other: &Span) -> bool {
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
    pub fn parent(self) -> Option<Span> {
        let expn_data = self.ctxt().outer_expn_data();
        if !expn_data.is_root() { Some(expn_data.call_site) } else { None }
    }

    /// Walk down the expansion ancestors to find a span that's contained within `outer`.
    pub fn find_ancestor_inside(mut self, outer: Span) -> Option<Span> {
        while !outer.contains(self) {
            self = self.parent()?;
        }
        Some(self)
    }

    /// Edition of the crate from which this span came.
    pub fn edition(self) -> edition::Edition {
        self.ctxt().edition()
    }

    #[inline]
    pub fn rust_2015(&self) -> bool {
        self.edition() == edition::Edition::Edition2015
    }

    #[inline]
    pub fn rust_2018(&self) -> bool {
        self.edition() >= edition::Edition::Edition2018
    }

    #[inline]
    pub fn rust_2021(&self) -> bool {
        self.edition() >= edition::Edition::Edition2021
    }

    /// Returns the source callee.
    ///
    /// Returns `None` if the supplied span has no expansion trace,
    /// else returns the `ExpnData` for the macro definition
    /// corresponding to the source callsite.
    pub fn source_callee(self) -> Option<ExpnData> {
        fn source_callee(expn_data: ExpnData) -> ExpnData {
            let next_expn_data = expn_data.call_site.ctxt().outer_expn_data();
            if !next_expn_data.is_root() { source_callee(next_expn_data) } else { expn_data }
        }
        let expn_data = self.ctxt().outer_expn_data();
        if !expn_data.is_root() { Some(source_callee(expn_data)) } else { None }
    }

    /// Checks if a span is "internal" to a macro in which `#[unstable]`
    /// items can be used (that is, a macro marked with
    /// `#[allow_internal_unstable]`).
    pub fn allows_unstable(&self, feature: Symbol) -> bool {
        self.ctxt()
            .outer_expn_data()
            .allow_internal_unstable
            .map_or(false, |features| features.iter().any(|&f| f == feature))
    }

    /// Checks if this span arises from a compiler desugaring of kind `kind`.
    pub fn is_desugaring(&self, kind: DesugaringKind) -> bool {
        match self.ctxt().outer_expn_data().kind {
            ExpnKind::Desugaring(k) => k == kind,
            _ => false,
        }
    }

    /// Returns the compiler desugaring that created this span, or `None`
    /// if this span is not from a desugaring.
    pub fn desugaring_kind(&self) -> Option<DesugaringKind> {
        match self.ctxt().outer_expn_data().kind {
            ExpnKind::Desugaring(k) => Some(k),
            _ => None,
        }
    }

    /// Checks if a span is "internal" to a macro in which `unsafe`
    /// can be used without triggering the `unsafe_code` lint.
    //  (that is, a macro marked with `#[allow_internal_unsafe]`).
    pub fn allows_unsafe(&self) -> bool {
        self.ctxt().outer_expn_data().allow_internal_unsafe
    }

    pub fn macro_backtrace(mut self) -> impl Iterator<Item = ExpnData> {
        let mut prev_span = DUMMY_SP;
        std::iter::from_fn(move || {
            loop {
                let expn_data = self.ctxt().outer_expn_data();
                if expn_data.is_root() {
                    return None;
                }

                let is_recursive = expn_data.call_site.source_equal(&prev_span);

                prev_span = self;
                self = expn_data.call_site;

                // Don't print recursive invocations.
                if !is_recursive {
                    return Some(expn_data);
                }
            }
        })
    }

    /// Returns a `Span` that would enclose both `self` and `end`.
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
            if span_data.ctxt == SyntaxContext::root() {
                return end;
            } else if end_data.ctxt == SyntaxContext::root() {
                return self;
            }
            // Both spans fall within a macro.
            // FIXME(estebank): check if it is the *same* macro.
        }
        Span::new(
            cmp::min(span_data.lo, end_data.lo),
            cmp::max(span_data.hi, end_data.hi),
            if span_data.ctxt == SyntaxContext::root() { end_data.ctxt } else { span_data.ctxt },
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
            if end.ctxt == SyntaxContext::root() { end.ctxt } else { span.ctxt },
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
        let span = self.data();
        let end = end.data();
        Span::new(
            span.lo,
            end.lo,
            if end.ctxt == SyntaxContext::root() { end.ctxt } else { span.ctxt },
        )
    }

    pub fn from_inner(self, inner: InnerSpan) -> Span {
        let span = self.data();
        Span::new(
            span.lo + BytePos::from_usize(inner.start),
            span.lo + BytePos::from_usize(inner.end),
            span.ctxt,
        )
    }

    /// Equivalent of `Span::def_site` from the proc macro API,
    /// except that the location is taken from the `self` span.
    pub fn with_def_site_ctxt(self, expn_id: ExpnId) -> Span {
        self.with_ctxt_from_mark(expn_id, Transparency::Opaque)
    }

    /// Equivalent of `Span::call_site` from the proc macro API,
    /// except that the location is taken from the `self` span.
    pub fn with_call_site_ctxt(&self, expn_id: ExpnId) -> Span {
        self.with_ctxt_from_mark(expn_id, Transparency::Transparent)
    }

    /// Equivalent of `Span::mixed_site` from the proc macro API,
    /// except that the location is taken from the `self` span.
    pub fn with_mixed_site_ctxt(&self, expn_id: ExpnId) -> Span {
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
        *self = Span::new(span.lo, span.hi, span.ctxt);
        mark
    }

    #[inline]
    pub fn adjust(&mut self, expn_id: ExpnId) -> Option<ExpnId> {
        let mut span = self.data();
        let mark = span.ctxt.adjust(expn_id);
        *self = Span::new(span.lo, span.hi, span.ctxt);
        mark
    }

    #[inline]
    pub fn normalize_to_macros_2_0_and_adjust(&mut self, expn_id: ExpnId) -> Option<ExpnId> {
        let mut span = self.data();
        let mark = span.ctxt.normalize_to_macros_2_0_and_adjust(expn_id);
        *self = Span::new(span.lo, span.hi, span.ctxt);
        mark
    }

    #[inline]
    pub fn glob_adjust(&mut self, expn_id: ExpnId, glob_span: Span) -> Option<Option<ExpnId>> {
        let mut span = self.data();
        let mark = span.ctxt.glob_adjust(expn_id, glob_span);
        *self = Span::new(span.lo, span.hi, span.ctxt);
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
        *self = Span::new(span.lo, span.hi, span.ctxt);
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

/// A span together with some additional data.
#[derive(Clone, Debug)]
pub struct SpanLabel {
    /// The span we are going to include in the final snippet.
    pub span: Span,

    /// Is this a primary span? This is the "locus" of the message,
    /// and is indicated with a `^^^^` underline, versus `----`.
    pub is_primary: bool,

    /// What label should we attach to this span (if any)?
    pub label: Option<String>,
}

impl Default for Span {
    fn default() -> Self {
        DUMMY_SP
    }
}

impl<E: Encoder> Encodable<E> for Span {
    default fn encode(&self, s: &mut E) -> Result<(), E::Error> {
        let span = self.data();
        s.emit_struct(false, |s| {
            s.emit_struct_field("lo", true, |s| span.lo.encode(s))?;
            s.emit_struct_field("hi", false, |s| span.hi.encode(s))
        })
    }
}
impl<D: Decoder> Decodable<D> for Span {
    default fn decode(s: &mut D) -> Result<Span, D::Error> {
        s.read_struct(|d| {
            let lo = d.read_struct_field("lo", Decodable::decode)?;
            let hi = d.read_struct_field("hi", Decodable::decode)?;

            Ok(Span::new(lo, hi, SyntaxContext::root()))
        })
    }
}

/// Calls the provided closure, using the provided `SourceMap` to format
/// any spans that are debug-printed during the closure's execution.
///
/// Normally, the global `TyCtxt` is used to retrieve the `SourceMap`
/// (see `rustc_interface::callbacks::span_debug1`). However, some parts
/// of the compiler (e.g. `rustc_parse`) may debug-print `Span`s before
/// a `TyCtxt` is available. In this case, we fall back to
/// the `SourceMap` provided to this function. If that is not available,
/// we fall back to printing the raw `Span` field values.
pub fn with_source_map<T, F: FnOnce() -> T>(source_map: Lrc<SourceMap>, f: F) -> T {
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

pub fn debug_with_source_map(
    span: Span,
    f: &mut fmt::Formatter<'_>,
    source_map: &SourceMap,
) -> fmt::Result {
    write!(f, "{} ({:?})", source_map.span_to_diagnostic_string(span), span.ctxt())
}

pub fn default_span_debug(span: Span, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    with_session_globals(|session_globals| {
        if let Some(source_map) = &*session_globals.source_map.borrow() {
            debug_with_source_map(span, f, source_map)
        } else {
            f.debug_struct("Span")
                .field("lo", &span.lo())
                .field("hi", &span.hi())
                .field("ctxt", &span.ctxt())
                .finish()
        }
    })
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (*SPAN_DEBUG)(*self, f)
    }
}

impl fmt::Debug for SpanData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (*SPAN_DEBUG)(Span::new(self.lo, self.hi, self.ctxt), f)
    }
}

impl MultiSpan {
    #[inline]
    pub fn new() -> MultiSpan {
        MultiSpan { primary_spans: vec![], span_labels: vec![] }
    }

    pub fn from_span(primary_span: Span) -> MultiSpan {
        MultiSpan { primary_spans: vec![primary_span], span_labels: vec![] }
    }

    pub fn from_spans(mut vec: Vec<Span>) -> MultiSpan {
        vec.sort();
        MultiSpan { primary_spans: vec, span_labels: vec![] }
    }

    pub fn push_span_label(&mut self, span: Span, label: String) {
        self.span_labels.push((span, label));
    }

    /// Selects the first primary span (if any).
    pub fn primary_span(&self) -> Option<Span> {
        self.primary_spans.first().cloned()
    }

    /// Returns all primary spans.
    pub fn primary_spans(&self) -> &[Span] {
        &self.primary_spans
    }

    /// Returns `true` if any of the primary spans are displayable.
    pub fn has_primary_spans(&self) -> bool {
        self.primary_spans.iter().any(|sp| !sp.is_dummy())
    }

    /// Returns `true` if this contains only a dummy primary span with any hygienic context.
    pub fn is_dummy(&self) -> bool {
        let mut is_dummy = true;
        for span in &self.primary_spans {
            if !span.is_dummy() {
                is_dummy = false;
            }
        }
        is_dummy
    }

    /// Replaces all occurrences of one Span with another. Used to move `Span`s in areas that don't
    /// display well (like std macros). Returns whether replacements occurred.
    pub fn replace(&mut self, before: Span, after: Span) -> bool {
        let mut replacements_occurred = false;
        for primary_span in &mut self.primary_spans {
            if *primary_span == before {
                *primary_span = after;
                replacements_occurred = true;
            }
        }
        for span_label in &mut self.span_labels {
            if span_label.0 == before {
                span_label.0 = after;
                replacements_occurred = true;
            }
        }
        replacements_occurred
    }

    /// Returns the strings to highlight. We always ensure that there
    /// is an entry for each of the primary spans -- for each primary
    /// span `P`, if there is at least one label with span `P`, we return
    /// those labels (marked as primary). But otherwise we return
    /// `SpanLabel` instances with empty labels.
    pub fn span_labels(&self) -> Vec<SpanLabel> {
        let is_primary = |span| self.primary_spans.contains(&span);

        let mut span_labels = self
            .span_labels
            .iter()
            .map(|&(span, ref label)| SpanLabel {
                span,
                is_primary: is_primary(span),
                label: Some(label.clone()),
            })
            .collect::<Vec<_>>();

        for &span in &self.primary_spans {
            if !span_labels.iter().any(|sl| sl.span == span) {
                span_labels.push(SpanLabel { span, is_primary: true, label: None });
            }
        }

        span_labels
    }

    /// Returns `true` if any of the span labels is displayable.
    pub fn has_span_labels(&self) -> bool {
        self.span_labels.iter().any(|(sp, _)| !sp.is_dummy())
    }
}

impl From<Span> for MultiSpan {
    fn from(span: Span) -> MultiSpan {
        MultiSpan::from_span(span)
    }
}

impl From<Vec<Span>> for MultiSpan {
    fn from(spans: Vec<Span>) -> MultiSpan {
        MultiSpan::from_spans(spans)
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
            _ => panic!("width {} given for non-narrow character", width),
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
        /// This SourceFile's byte-offset within the source_map of its original crate.
        original_start_pos: BytePos,
        /// The end of this SourceFile within the source_map of its original crate.
        original_end_pos: BytePos,
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

rustc_data_structures::impl_stable_hash_via_hash!(SourceFileHashAlgorithm);

/// The hash of the on-disk source file used for debug info.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
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

/// A single source in the [`SourceMap`].
#[derive(Clone)]
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
    pub lines: Vec<BytePos>,
    /// Locations of multi-byte characters in the source code.
    pub multibyte_chars: Vec<MultiByteChar>,
    /// Width of characters that are not narrow in the source code.
    pub non_narrow_chars: Vec<NonNarrowChar>,
    /// Locations of characters removed during normalization.
    pub normalized_pos: Vec<NormalizedPos>,
    /// A hash of the filename, used for speeding up hashing in incremental compilation.
    pub name_hash: u128,
    /// Indicates which crate this `SourceFile` was imported from.
    pub cnum: CrateNum,
}

impl<S: Encoder> Encodable<S> for SourceFile {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_struct(false, |s| {
            s.emit_struct_field("name", true, |s| self.name.encode(s))?;
            s.emit_struct_field("src_hash", false, |s| self.src_hash.encode(s))?;
            s.emit_struct_field("start_pos", false, |s| self.start_pos.encode(s))?;
            s.emit_struct_field("end_pos", false, |s| self.end_pos.encode(s))?;
            s.emit_struct_field("lines", false, |s| {
                let lines = &self.lines[..];
                // Store the length.
                s.emit_u32(lines.len() as u32)?;

                if !lines.is_empty() {
                    // In order to preserve some space, we exploit the fact that
                    // the lines list is sorted and individual lines are
                    // probably not that long. Because of that we can store lines
                    // as a difference list, using as little space as possible
                    // for the differences.
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

                    let bytes_per_diff: u8 = match max_line_length {
                        0..=0xFF => 1,
                        0x100..=0xFFFF => 2,
                        _ => 4,
                    };

                    // Encode the number of bytes used per diff.
                    bytes_per_diff.encode(s)?;

                    // Encode the first element.
                    lines[0].encode(s)?;

                    let diff_iter = lines[..].array_windows().map(|&[fst, snd]| snd - fst);

                    match bytes_per_diff {
                        1 => {
                            for diff in diff_iter {
                                (diff.0 as u8).encode(s)?
                            }
                        }
                        2 => {
                            for diff in diff_iter {
                                (diff.0 as u16).encode(s)?
                            }
                        }
                        4 => {
                            for diff in diff_iter {
                                diff.0.encode(s)?
                            }
                        }
                        _ => unreachable!(),
                    }
                }

                Ok(())
            })?;
            s.emit_struct_field("multibyte_chars", false, |s| self.multibyte_chars.encode(s))?;
            s.emit_struct_field("non_narrow_chars", false, |s| self.non_narrow_chars.encode(s))?;
            s.emit_struct_field("name_hash", false, |s| self.name_hash.encode(s))?;
            s.emit_struct_field("normalized_pos", false, |s| self.normalized_pos.encode(s))?;
            s.emit_struct_field("cnum", false, |s| self.cnum.encode(s))
        })
    }
}

impl<D: Decoder> Decodable<D> for SourceFile {
    fn decode(d: &mut D) -> Result<SourceFile, D::Error> {
        d.read_struct(|d| {
            let name: FileName = d.read_struct_field("name", |d| Decodable::decode(d))?;
            let src_hash: SourceFileHash =
                d.read_struct_field("src_hash", |d| Decodable::decode(d))?;
            let start_pos: BytePos = d.read_struct_field("start_pos", |d| Decodable::decode(d))?;
            let end_pos: BytePos = d.read_struct_field("end_pos", |d| Decodable::decode(d))?;
            let lines: Vec<BytePos> = d.read_struct_field("lines", |d| {
                let num_lines: u32 = Decodable::decode(d)?;
                let mut lines = Vec::with_capacity(num_lines as usize);

                if num_lines > 0 {
                    // Read the number of bytes used per diff.
                    let bytes_per_diff: u8 = Decodable::decode(d)?;

                    // Read the first element.
                    let mut line_start: BytePos = Decodable::decode(d)?;
                    lines.push(line_start);

                    for _ in 1..num_lines {
                        let diff = match bytes_per_diff {
                            1 => d.read_u8()? as u32,
                            2 => d.read_u16()? as u32,
                            4 => d.read_u32()?,
                            _ => unreachable!(),
                        };

                        line_start = line_start + BytePos(diff);

                        lines.push(line_start);
                    }
                }

                Ok(lines)
            })?;
            let multibyte_chars: Vec<MultiByteChar> =
                d.read_struct_field("multibyte_chars", |d| Decodable::decode(d))?;
            let non_narrow_chars: Vec<NonNarrowChar> =
                d.read_struct_field("non_narrow_chars", |d| Decodable::decode(d))?;
            let name_hash: u128 = d.read_struct_field("name_hash", |d| Decodable::decode(d))?;
            let normalized_pos: Vec<NormalizedPos> =
                d.read_struct_field("normalized_pos", |d| Decodable::decode(d))?;
            let cnum: CrateNum = d.read_struct_field("cnum", |d| Decodable::decode(d))?;
            Ok(SourceFile {
                name,
                start_pos,
                end_pos,
                src: None,
                src_hash,
                // Unused - the metadata decoder will construct
                // a new SourceFile, filling in `external_src` properly
                external_src: Lock::new(ExternalSource::Unneeded),
                lines,
                multibyte_chars,
                non_narrow_chars,
                normalized_pos,
                name_hash,
                cnum,
            })
        })
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
            hasher.finish::<u128>()
        };
        let end_pos = start_pos.to_usize() + src.len();
        assert!(end_pos <= u32::MAX as usize);

        let (lines, multibyte_chars, non_narrow_chars) =
            analyze_source_file::analyze_source_file(&src[..], start_pos);

        SourceFile {
            name,
            src: Some(Lrc::new(src)),
            src_hash,
            external_src: Lock::new(ExternalSource::Unneeded),
            start_pos,
            end_pos: Pos::from_usize(end_pos),
            lines,
            multibyte_chars,
            non_narrow_chars,
            normalized_pos,
            name_hash,
            cnum: LOCAL_CRATE,
        }
    }

    /// Returns the `BytePos` of the beginning of the current line.
    pub fn line_begin_pos(&self, pos: BytePos) -> BytePos {
        let line_index = self.lookup_line(pos).unwrap();
        self.lines[line_index]
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
            let line = self.lines.get(line_number)?;
            let begin: BytePos = *line - self.start_pos;
            begin.to_usize()
        };

        if let Some(ref src) = self.src {
            Some(Cow::from(get_until_newline(src, begin)))
        } else if let Some(src) = self.external_src.borrow().get_source() {
            Some(Cow::Owned(String::from(get_until_newline(src, begin))))
        } else {
            None
        }
    }

    pub fn is_real_file(&self) -> bool {
        self.name.is_real()
    }

    pub fn is_imported(&self) -> bool {
        self.src.is_none()
    }

    pub fn count_lines(&self) -> usize {
        self.lines.len()
    }

    /// Finds the line containing the given position. The return value is the
    /// index into the `lines` array of this `SourceFile`, not the 1-based line
    /// number. If the source_file is empty or the position is located before the
    /// first line, `None` is returned.
    pub fn lookup_line(&self, pos: BytePos) -> Option<usize> {
        match self.lines.binary_search(&pos) {
            Ok(idx) => Some(idx),
            Err(0) => None,
            Err(idx) => Some(idx - 1),
        }
    }

    pub fn line_bounds(&self, line_index: usize) -> Range<BytePos> {
        if self.is_empty() {
            return self.start_pos..self.end_pos;
        }

        assert!(line_index < self.lines.len());
        if line_index == (self.lines.len() - 1) {
            self.lines[line_index]..self.end_pos
        } else {
            self.lines[line_index]..self.lines[line_index + 1]
        }
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
                let linebpos = self.lines[a];
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
            let linebpos = self.lines[line - 1];
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

impl<S: rustc_serialize::Encoder> Encodable<S> for BytePos {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_u32(self.0)
    }
}

impl<D: rustc_serialize::Decoder> Decodable<D> for BytePos {
    fn decode(d: &mut D) -> Result<BytePos, D::Error> {
        Ok(BytePos(d.read_u32()?))
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

pub static SPAN_DEBUG: AtomicRef<fn(Span, &mut fmt::Formatter<'_>) -> fmt::Result> =
    AtomicRef::new(&(default_span_debug as fn(_, &mut fmt::Formatter<'_>) -> _));

// _____________________________________________________________________________
// SpanLinesError, SpanSnippetError, DistinctSources, MalformedSourceMapPositions
//

pub type FileLinesResult = Result<FileLines, SpanLinesError>;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SpanLinesError {
    DistinctSources(DistinctSources),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SpanSnippetError {
    IllFormedSpan(Span),
    DistinctSources(DistinctSources),
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
    fn span_data_to_lines_and_cols(
        &mut self,
        span: &SpanData,
    ) -> Option<(Lrc<SourceFile>, usize, BytePos, usize, BytePos)>;
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

        if !ctx.hash_spans() {
            return;
        }

        self.ctxt().hash_stable(ctx, hasher);

        if self.is_dummy() {
            Hash::hash(&TAG_INVALID_SPAN, hasher);
            return;
        }

        // If this is not an empty or invalid span, we want to hash the last
        // position that belongs to it, as opposed to hashing the first
        // position past it.
        let span = self.data();
        let (file, line_lo, col_lo, line_hi, col_hi) = match ctx.span_data_to_lines_and_cols(&span)
        {
            Some(pos) => pos,
            None => {
                Hash::hash(&TAG_INVALID_SPAN, hasher);
                return;
            }
        };

        Hash::hash(&TAG_VALID_SPAN, hasher);
        // We truncate the stable ID hash and line and column numbers. The chances
        // of causing a collision this way should be minimal.
        Hash::hash(&(file.name_hash as u64), hasher);

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
