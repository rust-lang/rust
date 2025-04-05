//! Source positions and related helper functions.
//!
//! Important concepts in this module include:
//!
//! - the *span*, represented by [`SpanData`] and related types;
//! - source code as represented by a [`SourceMap`]; and
//! - interned strings, represented by [`Symbol`]s, with some common symbols available statically
//!   in the [`sym`] module.
//!
//! Unlike most compilers, the span contains not only the position in the source code, but also
//! various other metadata, such as the edition and macro hygiene. This metadata is stored in
//! [`SyntaxContext`] and [`ExpnData`].
//!
//! ## Note
//!
//! This API is completely unstable and subject to change.

// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(array_windows)]
#![feature(cfg_match)]
#![feature(core_io_borrowed_buf)]
#![feature(hash_set_entry)]
#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(map_try_insert)]
#![feature(negative_impls)]
#![feature(read_buf)]
#![feature(round_char_boundary)]
#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]
#![feature(slice_as_chunks)]
// tidy-alphabetical-end

// The code produced by the `Encodable`/`Decodable` derive macros refer to
// `rustc_span::Span{Encoder,Decoder}`. That's fine outside this crate, but doesn't work inside
// this crate without this line making `rustc_span` available.
extern crate self as rustc_span;

use derive_where::derive_where;
use rustc_data_structures::{AtomicRef, outline};
use rustc_macros::{Decodable, Encodable, HashStable_Generic};
use rustc_serialize::opaque::{FileEncoder, MemDecoder};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use tracing::debug;

mod caching_source_map_view;
pub mod source_map;
use source_map::{SourceMap, SourceMapInputs};

pub use self::caching_source_map_view::CachingSourceMapView;
use crate::fatal_error::FatalError;

pub mod edition;
use edition::Edition;
pub mod hygiene;
use hygiene::Transparency;
pub use hygiene::{
    DesugaringKind, ExpnData, ExpnHash, ExpnId, ExpnKind, LocalExpnId, MacroKind, SyntaxContext,
};
use rustc_data_structures::stable_hasher::HashingControls;
pub mod def_id;
use def_id::{CrateNum, DefId, DefIndex, DefPathHash, LOCAL_CRATE, LocalDefId, StableCrateId};
pub mod edit_distance;
mod span_encoding;
pub use span_encoding::{DUMMY_SP, Span};

pub mod symbol;
pub use symbol::{Ident, MacroRulesNormalizedIdent, STDLIB_STABLE_CRATES, Symbol, kw, sym};

mod analyze_source_file;
pub mod fatal_error;

pub mod profiling;

use std::borrow::Cow;
use std::cmp::{self, Ordering};
use std::fmt::Display;
use std::hash::Hash;
use std::io::{self, Read};
use std::ops::{Add, Range, Sub};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::{fmt, iter};

use md5::{Digest, Md5};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::{FreezeLock, FreezeWriteGuard, Lock};
use rustc_data_structures::unord::UnordMap;
use rustc_hashes::{Hash64, Hash128};
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
    /// Maps a macro argument token into use of the corresponding metavariable in the macro body.
    /// Collisions are possible and processed in `maybe_use_metavar_location` on best effort basis.
    metavar_spans: MetavarSpansMap,
    hygiene_data: Lock<hygiene::HygieneData>,

    /// The session's source map, if there is one. This field should only be
    /// used in places where the `Session` is truly not available, such as
    /// `<Span as Debug>::fmt`.
    source_map: Option<Arc<SourceMap>>,
}

impl SessionGlobals {
    pub fn new(
        edition: Edition,
        extra_symbols: &[&'static str],
        sm_inputs: Option<SourceMapInputs>,
    ) -> SessionGlobals {
        SessionGlobals {
            symbol_interner: symbol::Interner::with_extra_symbols(extra_symbols),
            span_interner: Lock::new(span_encoding::SpanInterner::default()),
            metavar_spans: Default::default(),
            hygiene_data: Lock::new(hygiene::HygieneData::new(edition)),
            source_map: sm_inputs.map(|inputs| Arc::new(SourceMap::with_inputs(inputs))),
        }
    }
}

pub fn create_session_globals_then<R>(
    edition: Edition,
    extra_symbols: &[&'static str],
    sm_inputs: Option<SourceMapInputs>,
    f: impl FnOnce() -> R,
) -> R {
    assert!(
        !SESSION_GLOBALS.is_set(),
        "SESSION_GLOBALS should never be overwritten! \
         Use another thread if you need another SessionGlobals"
    );
    let session_globals = SessionGlobals::new(edition, extra_symbols, sm_inputs);
    SESSION_GLOBALS.set(&session_globals, f)
}

pub fn set_session_globals_then<R>(session_globals: &SessionGlobals, f: impl FnOnce() -> R) -> R {
    assert!(
        !SESSION_GLOBALS.is_set(),
        "SESSION_GLOBALS should never be overwritten! \
         Use another thread if you need another SessionGlobals"
    );
    SESSION_GLOBALS.set(session_globals, f)
}

/// No source map.
pub fn create_session_if_not_set_then<R, F>(edition: Edition, f: F) -> R
where
    F: FnOnce(&SessionGlobals) -> R,
{
    if !SESSION_GLOBALS.is_set() {
        let session_globals = SessionGlobals::new(edition, &[], None);
        SESSION_GLOBALS.set(&session_globals, || SESSION_GLOBALS.with(f))
    } else {
        SESSION_GLOBALS.with(f)
    }
}

pub fn with_session_globals<R, F>(f: F) -> R
where
    F: FnOnce(&SessionGlobals) -> R,
{
    SESSION_GLOBALS.with(f)
}

/// Default edition, no source map.
pub fn create_default_session_globals_then<R>(f: impl FnOnce() -> R) -> R {
    create_session_globals_then(edition::DEFAULT_EDITION, &[], None, f)
}

// If this ever becomes non thread-local, `decode_syntax_context`
// and `decode_expn_id` will need to be updated to handle concurrent
// deserialization.
scoped_tls::scoped_thread_local!(static SESSION_GLOBALS: SessionGlobals);

#[derive(Default)]
pub struct MetavarSpansMap(FreezeLock<UnordMap<Span, (Span, bool)>>);

impl MetavarSpansMap {
    pub fn insert(&self, span: Span, var_span: Span) -> bool {
        match self.0.write().try_insert(span, (var_span, false)) {
            Ok(_) => true,
            Err(entry) => entry.entry.get().0 == var_span,
        }
    }

    /// Read a span and record that it was read.
    pub fn get(&self, span: Span) -> Option<Span> {
        if let Some(mut mspans) = self.0.try_write() {
            if let Some((var_span, read)) = mspans.get_mut(&span) {
                *read = true;
                Some(*var_span)
            } else {
                None
            }
        } else {
            if let Some((span, true)) = self.0.read().get(&span) { Some(*span) } else { None }
        }
    }

    /// Freeze the set, and return the spans which have been read.
    ///
    /// After this is frozen, no spans that have not been read can be read.
    pub fn freeze_and_get_read_spans(&self) -> UnordMap<Span, Span> {
        self.0.freeze().items().filter(|(_, (_, b))| *b).map(|(s1, (s2, _))| (*s1, *s2)).collect()
    }
}

#[inline]
pub fn with_metavar_spans<R>(f: impl FnOnce(&MetavarSpansMap) -> R) -> R {
    with_session_globals(|session_globals| f(&session_globals.metavar_spans))
}

// FIXME: We should use this enum or something like it to get rid of the
// use of magic `/rust/1.x/...` paths across the board.
#[derive(Debug, Eq, PartialEq, Clone, Ord, PartialOrd, Decodable)]
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
            RealFileName::LocalPath(ref local_path) => {
                encoder.emit_u8(0);
                local_path.encode(encoder);
            }

            RealFileName::Remapped { ref local_path, ref virtual_name } => {
                encoder.emit_u8(1);
                // For privacy and build reproducibility, we must not embed host-dependant path
                // in artifacts if they have been remapped by --remap-path-prefix
                assert!(local_path.is_none());
                local_path.encode(encoder);
                virtual_name.encode(encoder);
            }
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

    /// Return the path remapped or not depending on the [`FileNameDisplayPreference`].
    ///
    /// For the purpose of this function, local and short preference are equal.
    pub fn to_path(&self, display_pref: FileNameDisplayPreference) -> &Path {
        match display_pref {
            FileNameDisplayPreference::Local | FileNameDisplayPreference::Short => {
                self.local_path_if_available()
            }
            FileNameDisplayPreference::Remapped => self.remapped_path_if_available(),
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
#[derive(Debug, Eq, PartialEq, Clone, Ord, PartialOrd, Hash, Decodable, Encodable)]
pub enum FileName {
    Real(RealFileName),
    /// Strings provided as `--cfg [cfgspec]`.
    CfgSpec(Hash64),
    /// Command line.
    Anon(Hash64),
    /// Hack in `src/librustc_ast/parse.rs`.
    // FIXME(jseyfried)
    MacroExpansion(Hash64),
    ProcMacroSourceCode(Hash64),
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
            CfgSpec(_) => write!(fmt, "<cfgspec>"),
            MacroExpansion(_) => write!(fmt, "<macro expansion>"),
            Anon(_) => write!(fmt, "<anon>"),
            ProcMacroSourceCode(_) => write!(fmt, "<proc-macro source code>"),
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
            FileName::Real(inner) => inner.to_string_lossy(self.display_pref),
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
            | CliCrateAttr(_)
            | Custom(_)
            | CfgSpec(_)
            | DocTest(_, _)
            | InlineAsm(_) => false,
        }
    }

    pub fn prefer_remapped_unconditionaly(&self) -> FileNameDisplay<'_> {
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
        FileName::CfgSpec(hasher.finish())
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

    /// Returns the path suitable for reading from the file system on the local host,
    /// if this information exists.
    /// Avoid embedding this in build artifacts; see `remapped_path_if_available()` for that.
    pub fn into_local_path(self) -> Option<PathBuf> {
        match self {
            FileName::Real(path) => path.into_local_path(),
            FileName::DocTest(path, _) => Some(path),
            _ => None,
        }
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
#[derive_where(PartialOrd, Ord)]
pub struct SpanData {
    pub lo: BytePos,
    pub hi: BytePos,
    /// Information about where the macro came from, if this piece of
    /// code was created by a macro expansion.
    #[derive_where(skip)]
    // `SyntaxContext` does not implement `Ord`.
    // The other fields are enough to determine in-file order.
    pub ctxt: SyntaxContext,
    #[derive_where(skip)]
    // `LocalDefId` does not implement `Ord`.
    // The other fields are enough to determine in-file order.
    pub parent: Option<LocalDefId>,
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
    /// Avoid if possible, `Span::map_ctxt` should be preferred.
    #[inline]
    fn with_ctxt(&self, ctxt: SyntaxContext) -> Span {
        Span::new(self.lo, self.hi, ctxt, self.parent)
    }
    /// Avoid if possible, `Span::with_parent` should be preferred.
    #[inline]
    fn with_parent(&self, parent: Option<LocalDefId>) -> Span {
        Span::new(self.lo, self.hi, self.ctxt, parent)
    }
    /// Returns `true` if this is a dummy span with any hygienic context.
    #[inline]
    pub fn is_dummy(self) -> bool {
        self.lo.0 == 0 && self.hi.0 == 0
    }
    /// Returns `true` if `self` fully encloses `other`.
    pub fn contains(self, other: Self) -> bool {
        self.lo <= other.lo && other.hi <= self.hi
    }
}

impl Default for SpanData {
    fn default() -> Self {
        Self { lo: BytePos(0), hi: BytePos(0), ctxt: SyntaxContext::root(), parent: None }
    }
}

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
    pub fn with_ctxt(self, ctxt: SyntaxContext) -> Span {
        self.map_ctxt(|_| ctxt)
    }

    #[inline]
    pub fn is_visible(self, sm: &SourceMap) -> bool {
        !self.is_dummy() && sm.is_span_accessible(self)
    }

    /// Returns whether `span` originates in a foreign crate's external macro.
    ///
    /// This is used to test whether a lint should not even begin to figure out whether it should
    /// be reported on the current node.
    pub fn in_external_macro(self, sm: &SourceMap) -> bool {
        let expn_data = self.ctxt().outer_expn_data();
        match expn_data.kind {
            ExpnKind::Root
            | ExpnKind::Desugaring(
                DesugaringKind::ForLoop
                | DesugaringKind::WhileLoop
                | DesugaringKind::OpaqueTy
                | DesugaringKind::Async
                | DesugaringKind::Await,
            ) => false,
            ExpnKind::AstPass(_) | ExpnKind::Desugaring(_) => true, // well, it's "external"
            ExpnKind::Macro(MacroKind::Bang, _) => {
                // Dummy span for the `def_site` means it's an external macro.
                expn_data.def_site.is_dummy() || sm.is_imported(expn_data.def_site)
            }
            ExpnKind::Macro { .. } => true, // definitely a plugin
        }
    }

    /// Returns `true` if `span` originates in a derive-macro's expansion.
    pub fn in_derive_expansion(self) -> bool {
        matches!(self.ctxt().outer_expn_data().kind, ExpnKind::Macro(MacroKind::Derive, _))
    }

    /// Return whether `span` is generated by `async` or `await`.
    pub fn is_from_async_await(self) -> bool {
        matches!(
            self.ctxt().outer_expn_data().kind,
            ExpnKind::Desugaring(DesugaringKind::Async | DesugaringKind::Await),
        )
    }

    /// Gate suggestions that would not be appropriate in a context the user didn't write.
    pub fn can_be_used_for_suggestions(self) -> bool {
        !self.from_expansion()
        // FIXME: If this span comes from a `derive` macro but it points at code the user wrote,
        // the callsite span and the span will be pointing at different places. It also means that
        // we can safely provide suggestions on this span.
            || (self.in_derive_expansion()
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

    /// Returns `true` if `self` touches or adjoins `other`.
    pub fn overlaps_or_adjacent(self, other: Span) -> bool {
        let span = self.data();
        let other = other.data();
        span.lo <= other.hi && other.lo <= span.hi
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

    /// Returns `Some(span)`, where the end is trimmed by the start of `other`.
    pub fn trim_end(self, other: Span) -> Option<Span> {
        let span = self.data();
        let other = other.data();
        if span.lo < other.lo { Some(span.with_hi(cmp::min(span.hi, other.lo))) } else { None }
    }

    /// Returns the source span -- this is either the supplied span, or the span for
    /// the macro callsite that expanded to it.
    pub fn source_callsite(self) -> Span {
        let ctxt = self.ctxt();
        if !ctxt.is_root() { ctxt.outer_expn_data().call_site.source_callsite() } else { self }
    }

    /// The `Span` for the tokens in the previous macro expansion from which `self` was generated,
    /// if any.
    pub fn parent_callsite(self) -> Option<Span> {
        let ctxt = self.ctxt();
        (!ctxt.is_root()).then(|| ctxt.outer_expn_data().call_site)
    }

    /// Walk down the expansion ancestors to find a span that's contained within `outer`.
    ///
    /// The span returned by this method may have a different [`SyntaxContext`] as `outer`.
    /// If you need to extend the span, use [`find_ancestor_inside_same_ctxt`] instead,
    /// because joining spans with different syntax contexts can create unexpected results.
    ///
    /// [`find_ancestor_inside_same_ctxt`]: Self::find_ancestor_inside_same_ctxt
    pub fn find_ancestor_inside(mut self, outer: Span) -> Option<Span> {
        while !outer.contains(self) {
            self = self.parent_callsite()?;
        }
        Some(self)
    }

    /// Walk down the expansion ancestors to find a span with the same [`SyntaxContext`] as
    /// `other`.
    ///
    /// Like [`find_ancestor_inside_same_ctxt`], but specifically for when spans might not
    /// overlap. Take care when using this, and prefer [`find_ancestor_inside`] or
    /// [`find_ancestor_inside_same_ctxt`] when you know that the spans are nested (modulo
    /// macro expansion).
    ///
    /// [`find_ancestor_inside`]: Self::find_ancestor_inside
    /// [`find_ancestor_inside_same_ctxt`]: Self::find_ancestor_inside_same_ctxt
    pub fn find_ancestor_in_same_ctxt(mut self, other: Span) -> Option<Span> {
        while !self.eq_ctxt(other) {
            self = self.parent_callsite()?;
        }
        Some(self)
    }

    /// Walk down the expansion ancestors to find a span that's contained within `outer` and
    /// has the same [`SyntaxContext`] as `outer`.
    ///
    /// This method is the combination of [`find_ancestor_inside`] and
    /// [`find_ancestor_in_same_ctxt`] and should be preferred when extending the returned span.
    /// If you do not need to modify the span, use [`find_ancestor_inside`] instead.
    ///
    /// [`find_ancestor_inside`]: Self::find_ancestor_inside
    /// [`find_ancestor_in_same_ctxt`]: Self::find_ancestor_in_same_ctxt
    pub fn find_ancestor_inside_same_ctxt(mut self, outer: Span) -> Option<Span> {
        while !outer.contains(self) || !self.eq_ctxt(outer) {
            self = self.parent_callsite()?;
        }
        Some(self)
    }

    /// Recursively walk down the expansion ancestors to find the oldest ancestor span with the same
    /// [`SyntaxContext`] the initial span.
    ///
    /// This method is suitable for peeling through *local* macro expansions to find the "innermost"
    /// span that is still local and shares the same [`SyntaxContext`]. For example, given
    ///
    /// ```ignore (illustrative example, contains type error)
    ///  macro_rules! outer {
    ///      ($x: expr) => {
    ///          inner!($x)
    ///      }
    ///  }
    ///
    ///  macro_rules! inner {
    ///      ($x: expr) => {
    ///          format!("error: {}", $x)
    ///          //~^ ERROR mismatched types
    ///      }
    ///  }
    ///
    ///  fn bar(x: &str) -> Result<(), Box<dyn std::error::Error>> {
    ///      Err(outer!(x))
    ///  }
    /// ```
    ///
    /// if provided the initial span of `outer!(x)` inside `bar`, this method will recurse
    /// the parent callsites until we reach `format!("error: {}", $x)`, at which point it is the
    /// oldest ancestor span that is both still local and shares the same [`SyntaxContext`] as the
    /// initial span.
    pub fn find_oldest_ancestor_in_same_ctxt(self) -> Span {
        let mut cur = self;
        while cur.eq_ctxt(self)
            && let Some(parent_callsite) = cur.parent_callsite()
        {
            cur = parent_callsite;
        }
        cur
    }

    /// Edition of the crate from which this span came.
    pub fn edition(self) -> edition::Edition {
        self.ctxt().edition()
    }

    /// Is this edition 2015?
    #[inline]
    pub fn is_rust_2015(self) -> bool {
        self.edition().is_rust_2015()
    }

    /// Are we allowed to use features from the Rust 2018 edition?
    #[inline]
    pub fn at_least_rust_2018(self) -> bool {
        self.edition().at_least_rust_2018()
    }

    /// Are we allowed to use features from the Rust 2021 edition?
    #[inline]
    pub fn at_least_rust_2021(self) -> bool {
        self.edition().at_least_rust_2021()
    }

    /// Are we allowed to use features from the Rust 2024 edition?
    #[inline]
    pub fn at_least_rust_2024(self) -> bool {
        self.edition().at_least_rust_2024()
    }

    /// Returns the source callee.
    ///
    /// Returns `None` if the supplied span has no expansion trace,
    /// else returns the `ExpnData` for the macro definition
    /// corresponding to the source callsite.
    pub fn source_callee(self) -> Option<ExpnData> {
        let mut ctxt = self.ctxt();
        let mut opt_expn_data = None;
        while !ctxt.is_root() {
            let expn_data = ctxt.outer_expn_data();
            ctxt = expn_data.call_site.ctxt();
            opt_expn_data = Some(expn_data);
        }
        opt_expn_data
    }

    /// Checks if a span is "internal" to a macro in which `#[unstable]`
    /// items can be used (that is, a macro marked with
    /// `#[allow_internal_unstable]`).
    pub fn allows_unstable(self, feature: Symbol) -> bool {
        self.ctxt()
            .outer_expn_data()
            .allow_internal_unstable
            .is_some_and(|features| features.contains(&feature))
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
                let ctxt = self.ctxt();
                if ctxt.is_root() {
                    return None;
                }

                let expn_data = ctxt.outer_expn_data();
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

    /// Check if you can select metavar spans for the given spans to get matching contexts.
    fn try_metavars(a: SpanData, b: SpanData, a_orig: Span, b_orig: Span) -> (SpanData, SpanData) {
        match with_metavar_spans(|mspans| (mspans.get(a_orig), mspans.get(b_orig))) {
            (None, None) => {}
            (Some(meta_a), None) => {
                let meta_a = meta_a.data();
                if meta_a.ctxt == b.ctxt {
                    return (meta_a, b);
                }
            }
            (None, Some(meta_b)) => {
                let meta_b = meta_b.data();
                if a.ctxt == meta_b.ctxt {
                    return (a, meta_b);
                }
            }
            (Some(meta_a), Some(meta_b)) => {
                let meta_b = meta_b.data();
                if a.ctxt == meta_b.ctxt {
                    return (a, meta_b);
                }
                let meta_a = meta_a.data();
                if meta_a.ctxt == b.ctxt {
                    return (meta_a, b);
                } else if meta_a.ctxt == meta_b.ctxt {
                    return (meta_a, meta_b);
                }
            }
        }

        (a, b)
    }

    /// Prepare two spans to a combine operation like `to` or `between`.
    fn prepare_to_combine(
        a_orig: Span,
        b_orig: Span,
    ) -> Result<(SpanData, SpanData, Option<LocalDefId>), Span> {
        let (a, b) = (a_orig.data(), b_orig.data());
        if a.ctxt == b.ctxt {
            return Ok((a, b, if a.parent == b.parent { a.parent } else { None }));
        }

        let (a, b) = Span::try_metavars(a, b, a_orig, b_orig);
        if a.ctxt == b.ctxt {
            return Ok((a, b, if a.parent == b.parent { a.parent } else { None }));
        }

        // Context mismatches usually happen when procedural macros combine spans copied from
        // the macro input with spans produced by the macro (`Span::*_site`).
        // In that case we consider the combined span to be produced by the macro and return
        // the original macro-produced span as the result.
        // Otherwise we just fall back to returning the first span.
        // Combining locations typically doesn't make sense in case of context mismatches.
        // `is_root` here is a fast path optimization.
        let a_is_callsite = a.ctxt.is_root() || a.ctxt == b.span().source_callsite().ctxt();
        Err(if a_is_callsite { b_orig } else { a_orig })
    }

    /// This span, but in a larger context, may switch to the metavariable span if suitable.
    pub fn with_neighbor(self, neighbor: Span) -> Span {
        match Span::prepare_to_combine(self, neighbor) {
            Ok((this, ..)) => this.span(),
            Err(_) => self,
        }
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
        match Span::prepare_to_combine(self, end) {
            Ok((from, to, parent)) => {
                Span::new(cmp::min(from.lo, to.lo), cmp::max(from.hi, to.hi), from.ctxt, parent)
            }
            Err(fallback) => fallback,
        }
    }

    /// Returns a `Span` between the end of `self` to the beginning of `end`.
    ///
    /// ```text
    ///     ____             ___
    ///     self lorem ipsum end
    ///         ^^^^^^^^^^^^^
    /// ```
    pub fn between(self, end: Span) -> Span {
        match Span::prepare_to_combine(self, end) {
            Ok((from, to, parent)) => {
                Span::new(cmp::min(from.hi, to.hi), cmp::max(from.lo, to.lo), from.ctxt, parent)
            }
            Err(fallback) => fallback,
        }
    }

    /// Returns a `Span` from the beginning of `self` until the beginning of `end`.
    ///
    /// ```text
    ///     ____             ___
    ///     self lorem ipsum end
    ///     ^^^^^^^^^^^^^^^^^
    /// ```
    pub fn until(self, end: Span) -> Span {
        match Span::prepare_to_combine(self, end) {
            Ok((from, to, parent)) => {
                Span::new(cmp::min(from.lo, to.lo), cmp::max(from.lo, to.lo), from.ctxt, parent)
            }
            Err(fallback) => fallback,
        }
    }

    /// Returns the `Span` within the syntax context of "within". This is useful when
    /// "self" is an expansion from a macro variable, since this can be used for
    /// providing extra macro expansion context for certain errors.
    ///
    /// ```text
    /// macro_rules! m {
    ///     ($ident:ident) => { ($ident,) }
    /// }
    ///
    /// m!(outer_ident);
    /// ```
    ///
    /// If "self" is the span of the outer_ident, and "within" is the span of the `($ident,)`
    /// expr, then this will return the span of the `$ident` macro variable.
    pub fn within_macro(self, within: Span, sm: &SourceMap) -> Option<Span> {
        match Span::prepare_to_combine(self, within) {
            // Only return something if it doesn't overlap with the original span,
            // and the span isn't "imported" (i.e. from unavailable sources).
            // FIXME: This does limit the usefulness of the error when the macro is
            // from a foreign crate; we could also take into account `-Zmacro-backtrace`,
            // which doesn't redact this span (but that would mean passing in even more
            // args to this function, lol).
            Ok((self_, _, parent))
                if self_.hi < self.lo() || self.hi() < self_.lo && !sm.is_imported(within) =>
            {
                Some(Span::new(self_.lo, self_.hi, self_.ctxt, parent))
            }
            _ => None,
        }
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
    fn with_ctxt_from_mark(self, expn_id: ExpnId, transparency: Transparency) -> Span {
        self.with_ctxt(SyntaxContext::root().apply_mark(expn_id, transparency))
    }

    #[inline]
    pub fn apply_mark(self, expn_id: ExpnId, transparency: Transparency) -> Span {
        self.map_ctxt(|ctxt| ctxt.apply_mark(expn_id, transparency))
    }

    #[inline]
    pub fn remove_mark(&mut self) -> ExpnId {
        let mut mark = ExpnId::root();
        *self = self.map_ctxt(|mut ctxt| {
            mark = ctxt.remove_mark();
            ctxt
        });
        mark
    }

    #[inline]
    pub fn adjust(&mut self, expn_id: ExpnId) -> Option<ExpnId> {
        let mut mark = None;
        *self = self.map_ctxt(|mut ctxt| {
            mark = ctxt.adjust(expn_id);
            ctxt
        });
        mark
    }

    #[inline]
    pub fn normalize_to_macros_2_0_and_adjust(&mut self, expn_id: ExpnId) -> Option<ExpnId> {
        let mut mark = None;
        *self = self.map_ctxt(|mut ctxt| {
            mark = ctxt.normalize_to_macros_2_0_and_adjust(expn_id);
            ctxt
        });
        mark
    }

    #[inline]
    pub fn glob_adjust(&mut self, expn_id: ExpnId, glob_span: Span) -> Option<Option<ExpnId>> {
        let mut mark = None;
        *self = self.map_ctxt(|mut ctxt| {
            mark = ctxt.glob_adjust(expn_id, glob_span);
            ctxt
        });
        mark
    }

    #[inline]
    pub fn reverse_glob_adjust(
        &mut self,
        expn_id: ExpnId,
        glob_span: Span,
    ) -> Option<Option<ExpnId>> {
        let mut mark = None;
        *self = self.map_ctxt(|mut ctxt| {
            mark = ctxt.reverse_glob_adjust(expn_id, glob_span);
            ctxt
        });
        mark
    }

    #[inline]
    pub fn normalize_to_macros_2_0(self) -> Span {
        self.map_ctxt(|ctxt| ctxt.normalize_to_macros_2_0())
    }

    #[inline]
    pub fn normalize_to_macro_rules(self) -> Span {
        self.map_ctxt(|ctxt| ctxt.normalize_to_macro_rules())
    }
}

impl Default for Span {
    fn default() -> Self {
        DUMMY_SP
    }
}

rustc_index::newtype_index! {
    #[orderable]
    #[debug_format = "AttrId({})"]
    pub struct AttrId {}
}

/// This trait is used to allow encoder specific encodings of certain types.
/// It is similar to rustc_type_ir's TyEncoder.
pub trait SpanEncoder: Encoder {
    fn encode_span(&mut self, span: Span);
    fn encode_symbol(&mut self, symbol: Symbol);
    fn encode_expn_id(&mut self, expn_id: ExpnId);
    fn encode_syntax_context(&mut self, syntax_context: SyntaxContext);
    /// As a local identifier, a `CrateNum` is only meaningful within its context, e.g. within a tcx.
    /// Therefore, make sure to include the context when encode a `CrateNum`.
    fn encode_crate_num(&mut self, crate_num: CrateNum);
    fn encode_def_index(&mut self, def_index: DefIndex);
    fn encode_def_id(&mut self, def_id: DefId);
}

impl SpanEncoder for FileEncoder {
    fn encode_span(&mut self, span: Span) {
        let span = span.data();
        span.lo.encode(self);
        span.hi.encode(self);
    }

    fn encode_symbol(&mut self, symbol: Symbol) {
        self.emit_str(symbol.as_str());
    }

    fn encode_expn_id(&mut self, _expn_id: ExpnId) {
        panic!("cannot encode `ExpnId` with `FileEncoder`");
    }

    fn encode_syntax_context(&mut self, _syntax_context: SyntaxContext) {
        panic!("cannot encode `SyntaxContext` with `FileEncoder`");
    }

    fn encode_crate_num(&mut self, crate_num: CrateNum) {
        self.emit_u32(crate_num.as_u32());
    }

    fn encode_def_index(&mut self, _def_index: DefIndex) {
        panic!("cannot encode `DefIndex` with `FileEncoder`");
    }

    fn encode_def_id(&mut self, def_id: DefId) {
        def_id.krate.encode(self);
        def_id.index.encode(self);
    }
}

impl<E: SpanEncoder> Encodable<E> for Span {
    fn encode(&self, s: &mut E) {
        s.encode_span(*self);
    }
}

impl<E: SpanEncoder> Encodable<E> for Symbol {
    fn encode(&self, s: &mut E) {
        s.encode_symbol(*self);
    }
}

impl<E: SpanEncoder> Encodable<E> for ExpnId {
    fn encode(&self, s: &mut E) {
        s.encode_expn_id(*self)
    }
}

impl<E: SpanEncoder> Encodable<E> for SyntaxContext {
    fn encode(&self, s: &mut E) {
        s.encode_syntax_context(*self)
    }
}

impl<E: SpanEncoder> Encodable<E> for CrateNum {
    fn encode(&self, s: &mut E) {
        s.encode_crate_num(*self)
    }
}

impl<E: SpanEncoder> Encodable<E> for DefIndex {
    fn encode(&self, s: &mut E) {
        s.encode_def_index(*self)
    }
}

impl<E: SpanEncoder> Encodable<E> for DefId {
    fn encode(&self, s: &mut E) {
        s.encode_def_id(*self)
    }
}

impl<E: SpanEncoder> Encodable<E> for AttrId {
    fn encode(&self, _s: &mut E) {
        // A fresh id will be generated when decoding
    }
}

/// This trait is used to allow decoder specific encodings of certain types.
/// It is similar to rustc_type_ir's TyDecoder.
pub trait SpanDecoder: Decoder {
    fn decode_span(&mut self) -> Span;
    fn decode_symbol(&mut self) -> Symbol;
    fn decode_expn_id(&mut self) -> ExpnId;
    fn decode_syntax_context(&mut self) -> SyntaxContext;
    fn decode_crate_num(&mut self) -> CrateNum;
    fn decode_def_index(&mut self) -> DefIndex;
    fn decode_def_id(&mut self) -> DefId;
    fn decode_attr_id(&mut self) -> AttrId;
}

impl SpanDecoder for MemDecoder<'_> {
    fn decode_span(&mut self) -> Span {
        let lo = Decodable::decode(self);
        let hi = Decodable::decode(self);

        Span::new(lo, hi, SyntaxContext::root(), None)
    }

    fn decode_symbol(&mut self) -> Symbol {
        Symbol::intern(self.read_str())
    }

    fn decode_expn_id(&mut self) -> ExpnId {
        panic!("cannot decode `ExpnId` with `MemDecoder`");
    }

    fn decode_syntax_context(&mut self) -> SyntaxContext {
        panic!("cannot decode `SyntaxContext` with `MemDecoder`");
    }

    fn decode_crate_num(&mut self) -> CrateNum {
        CrateNum::from_u32(self.read_u32())
    }

    fn decode_def_index(&mut self) -> DefIndex {
        panic!("cannot decode `DefIndex` with `MemDecoder`");
    }

    fn decode_def_id(&mut self) -> DefId {
        DefId { krate: Decodable::decode(self), index: Decodable::decode(self) }
    }

    fn decode_attr_id(&mut self) -> AttrId {
        panic!("cannot decode `AttrId` with `MemDecoder`");
    }
}

impl<D: SpanDecoder> Decodable<D> for Span {
    fn decode(s: &mut D) -> Span {
        s.decode_span()
    }
}

impl<D: SpanDecoder> Decodable<D> for Symbol {
    fn decode(s: &mut D) -> Symbol {
        s.decode_symbol()
    }
}

impl<D: SpanDecoder> Decodable<D> for ExpnId {
    fn decode(s: &mut D) -> ExpnId {
        s.decode_expn_id()
    }
}

impl<D: SpanDecoder> Decodable<D> for SyntaxContext {
    fn decode(s: &mut D) -> SyntaxContext {
        s.decode_syntax_context()
    }
}

impl<D: SpanDecoder> Decodable<D> for CrateNum {
    fn decode(s: &mut D) -> CrateNum {
        s.decode_crate_num()
    }
}

impl<D: SpanDecoder> Decodable<D> for DefIndex {
    fn decode(s: &mut D) -> DefIndex {
        s.decode_def_index()
    }
}

impl<D: SpanDecoder> Decodable<D> for DefId {
    fn decode(s: &mut D) -> DefId {
        s.decode_def_id()
    }
}

impl<D: SpanDecoder> Decodable<D> for AttrId {
    fn decode(s: &mut D) -> AttrId {
        s.decode_attr_id()
    }
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
                if let Some(source_map) = &session_globals.source_map {
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
        fmt::Debug::fmt(&self.span(), f)
    }
}

/// Identifies an offset of a multi-byte character in a `SourceFile`.
#[derive(Copy, Clone, Encodable, Decodable, Eq, PartialEq, Debug, HashStable_Generic)]
pub struct MultiByteChar {
    /// The relative offset of the character in the `SourceFile`.
    pub pos: RelativeBytePos,
    /// The number of bytes, `>= 2`.
    pub bytes: u8,
}

/// Identifies an offset of a character that was normalized away from `SourceFile`.
#[derive(Copy, Clone, Encodable, Decodable, Eq, PartialEq, Debug, HashStable_Generic)]
pub struct NormalizedPos {
    /// The relative offset of the character in the `SourceFile`.
    pub pos: RelativeBytePos,
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
    Present(Arc<String>),
    /// No attempt has been made to load the external source.
    AbsentOk,
    /// A failed attempt has been made to load the external source.
    AbsentErr,
}

impl ExternalSource {
    pub fn get_source(&self) -> Option<&str> {
        match self {
            ExternalSource::Foreign { kind: ExternalSourceKind::Present(src), .. } => Some(src),
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
    Blake3,
}

impl Display for SourceFileHashAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Md5 => "md5",
            Self::Sha1 => "sha1",
            Self::Sha256 => "sha256",
            Self::Blake3 => "blake3",
        })
    }
}

impl FromStr for SourceFileHashAlgorithm {
    type Err = ();

    fn from_str(s: &str) -> Result<SourceFileHashAlgorithm, ()> {
        match s {
            "md5" => Ok(SourceFileHashAlgorithm::Md5),
            "sha1" => Ok(SourceFileHashAlgorithm::Sha1),
            "sha256" => Ok(SourceFileHashAlgorithm::Sha256),
            "blake3" => Ok(SourceFileHashAlgorithm::Blake3),
            _ => Err(()),
        }
    }
}

/// The hash of the on-disk source file used for debug info and cargo freshness checks.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[derive(HashStable_Generic, Encodable, Decodable)]
pub struct SourceFileHash {
    pub kind: SourceFileHashAlgorithm,
    value: [u8; 32],
}

impl Display for SourceFileHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}=", self.kind)?;
        for byte in self.value[0..self.hash_len()].into_iter() {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl SourceFileHash {
    pub fn new_in_memory(kind: SourceFileHashAlgorithm, src: impl AsRef<[u8]>) -> SourceFileHash {
        let mut hash = SourceFileHash { kind, value: Default::default() };
        let len = hash.hash_len();
        let value = &mut hash.value[..len];
        let data = src.as_ref();
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
            SourceFileHashAlgorithm::Blake3 => value.copy_from_slice(blake3::hash(data).as_bytes()),
        };
        hash
    }

    pub fn new(kind: SourceFileHashAlgorithm, src: impl Read) -> Result<SourceFileHash, io::Error> {
        let mut hash = SourceFileHash { kind, value: Default::default() };
        let len = hash.hash_len();
        let value = &mut hash.value[..len];
        // Buffer size is the recommended amount to fully leverage SIMD instructions on AVX-512 as per
        // blake3 documentation.
        let mut buf = vec![0; 16 * 1024];

        fn digest<T>(
            mut hasher: T,
            mut update: impl FnMut(&mut T, &[u8]),
            finish: impl FnOnce(T, &mut [u8]),
            mut src: impl Read,
            buf: &mut [u8],
            value: &mut [u8],
        ) -> Result<(), io::Error> {
            loop {
                let bytes_read = src.read(buf)?;
                if bytes_read == 0 {
                    break;
                }
                update(&mut hasher, &buf[0..bytes_read]);
            }
            finish(hasher, value);
            Ok(())
        }

        match kind {
            SourceFileHashAlgorithm::Sha256 => {
                digest(
                    Sha256::new(),
                    |h, b| {
                        h.update(b);
                    },
                    |h, out| out.copy_from_slice(&h.finalize()),
                    src,
                    &mut buf,
                    value,
                )?;
            }
            SourceFileHashAlgorithm::Sha1 => {
                digest(
                    Sha1::new(),
                    |h, b| {
                        h.update(b);
                    },
                    |h, out| out.copy_from_slice(&h.finalize()),
                    src,
                    &mut buf,
                    value,
                )?;
            }
            SourceFileHashAlgorithm::Md5 => {
                digest(
                    Md5::new(),
                    |h, b| {
                        h.update(b);
                    },
                    |h, out| out.copy_from_slice(&h.finalize()),
                    src,
                    &mut buf,
                    value,
                )?;
            }
            SourceFileHashAlgorithm::Blake3 => {
                digest(
                    blake3::Hasher::new(),
                    |h, b| {
                        h.update(b);
                    },
                    |h, out| out.copy_from_slice(h.finalize().as_bytes()),
                    src,
                    &mut buf,
                    value,
                )?;
            }
        }
        Ok(hash)
    }

    /// Check if the stored hash matches the hash of the string.
    pub fn matches(&self, src: &str) -> bool {
        Self::new_in_memory(self.kind, src.as_bytes()) == *self
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
            SourceFileHashAlgorithm::Sha256 | SourceFileHashAlgorithm::Blake3 => 32,
        }
    }
}

#[derive(Clone)]
pub enum SourceFileLines {
    /// The source file lines, in decoded (random-access) form.
    Lines(Vec<RelativeBytePos>),

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
    pub src: Option<Arc<String>>,
    /// The source code's hash.
    pub src_hash: SourceFileHash,
    /// Used to enable cargo to use checksums to check if a crate is fresh rather
    /// than mtimes. This might be the same as `src_hash`, and if the requested algorithm
    /// is identical we won't compute it twice.
    pub checksum_hash: Option<SourceFileHash>,
    /// The external source code (used for external crates, which will have a `None`
    /// value as `self.src`.
    pub external_src: FreezeLock<ExternalSource>,
    /// The start position of this source in the `SourceMap`.
    pub start_pos: BytePos,
    /// The byte length of this source.
    pub source_len: RelativeBytePos,
    /// Locations of lines beginnings in the source code.
    pub lines: FreezeLock<SourceFileLines>,
    /// Locations of multi-byte characters in the source code.
    pub multibyte_chars: Vec<MultiByteChar>,
    /// Locations of characters removed during normalization.
    pub normalized_pos: Vec<NormalizedPos>,
    /// A hash of the filename & crate-id, used for uniquely identifying source
    /// files within the crate graph and for speeding up hashing in incremental
    /// compilation.
    pub stable_id: StableSourceFileId,
    /// Indicates which crate this `SourceFile` was imported from.
    pub cnum: CrateNum,
}

impl Clone for SourceFile {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            src: self.src.clone(),
            src_hash: self.src_hash,
            checksum_hash: self.checksum_hash,
            external_src: self.external_src.clone(),
            start_pos: self.start_pos,
            source_len: self.source_len,
            lines: self.lines.clone(),
            multibyte_chars: self.multibyte_chars.clone(),
            normalized_pos: self.normalized_pos.clone(),
            stable_id: self.stable_id,
            cnum: self.cnum,
        }
    }
}

impl<S: SpanEncoder> Encodable<S> for SourceFile {
    fn encode(&self, s: &mut S) {
        self.name.encode(s);
        self.src_hash.encode(s);
        self.checksum_hash.encode(s);
        // Do not encode `start_pos` as it's global state for this session.
        self.source_len.encode(s);

        // We are always in `Lines` form by the time we reach here.
        assert!(self.lines.read().is_lines());
        let lines = self.lines();
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
            assert_eq!(lines[0], RelativeBytePos(0));

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

        self.multibyte_chars.encode(s);
        self.stable_id.encode(s);
        self.normalized_pos.encode(s);
        self.cnum.encode(s);
    }
}

impl<D: SpanDecoder> Decodable<D> for SourceFile {
    fn decode(d: &mut D) -> SourceFile {
        let name: FileName = Decodable::decode(d);
        let src_hash: SourceFileHash = Decodable::decode(d);
        let checksum_hash: Option<SourceFileHash> = Decodable::decode(d);
        let source_len: RelativeBytePos = Decodable::decode(d);
        let lines = {
            let num_lines: u32 = Decodable::decode(d);
            if num_lines > 0 {
                // Read the number of bytes used per diff.
                let bytes_per_diff = d.read_u8() as usize;

                // Read the difference list.
                let num_diffs = num_lines as usize - 1;
                let raw_diffs = d.read_raw_bytes(bytes_per_diff * num_diffs).to_vec();
                SourceFileLines::Diffs(SourceFileDiffs { bytes_per_diff, num_diffs, raw_diffs })
            } else {
                SourceFileLines::Lines(vec![])
            }
        };
        let multibyte_chars: Vec<MultiByteChar> = Decodable::decode(d);
        let stable_id = Decodable::decode(d);
        let normalized_pos: Vec<NormalizedPos> = Decodable::decode(d);
        let cnum: CrateNum = Decodable::decode(d);
        SourceFile {
            name,
            start_pos: BytePos::from_u32(0),
            source_len,
            src: None,
            src_hash,
            checksum_hash,
            // Unused - the metadata decoder will construct
            // a new SourceFile, filling in `external_src` properly
            external_src: FreezeLock::frozen(ExternalSource::Unneeded),
            lines: FreezeLock::new(lines),
            multibyte_chars,
            normalized_pos,
            stable_id,
            cnum,
        }
    }
}

impl fmt::Debug for SourceFile {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "SourceFile({:?})", self.name)
    }
}

/// This is a [SourceFile] identifier that is used to correlate source files between
/// subsequent compilation sessions (which is something we need to do during
/// incremental compilation).
///
/// It is a hash value (so we can efficiently consume it when stable-hashing
/// spans) that consists of the `FileName` and the `StableCrateId` of the crate
/// the source file is from. The crate id is needed because sometimes the
/// `FileName` is not unique within the crate graph (think `src/lib.rs`, for
/// example).
///
/// The way the crate-id part is handled is a bit special: source files of the
/// local crate are hashed as `(filename, None)`, while source files from
/// upstream crates have a hash of `(filename, Some(stable_crate_id))`. This
/// is because SourceFiles for the local crate are allocated very early in the
/// compilation process when the `StableCrateId` is not yet known. If, due to
/// some refactoring of the compiler, the `StableCrateId` of the local crate
/// were to become available, it would be better to uniformly make this a
/// hash of `(filename, stable_crate_id)`.
///
/// When `SourceFile`s are exported in crate metadata, the `StableSourceFileId`
/// is updated to incorporate the `StableCrateId` of the exporting crate.
#[derive(
    Debug,
    Clone,
    Copy,
    Hash,
    PartialEq,
    Eq,
    HashStable_Generic,
    Encodable,
    Decodable,
    Default,
    PartialOrd,
    Ord
)]
pub struct StableSourceFileId(Hash128);

impl StableSourceFileId {
    fn from_filename_in_current_crate(filename: &FileName) -> Self {
        Self::from_filename_and_stable_crate_id(filename, None)
    }

    pub fn from_filename_for_export(
        filename: &FileName,
        local_crate_stable_crate_id: StableCrateId,
    ) -> Self {
        Self::from_filename_and_stable_crate_id(filename, Some(local_crate_stable_crate_id))
    }

    fn from_filename_and_stable_crate_id(
        filename: &FileName,
        stable_crate_id: Option<StableCrateId>,
    ) -> Self {
        let mut hasher = StableHasher::new();
        filename.hash(&mut hasher);
        stable_crate_id.hash(&mut hasher);
        StableSourceFileId(hasher.finish())
    }
}

impl SourceFile {
    const MAX_FILE_SIZE: u32 = u32::MAX - 1;

    pub fn new(
        name: FileName,
        mut src: String,
        hash_kind: SourceFileHashAlgorithm,
        checksum_hash_kind: Option<SourceFileHashAlgorithm>,
    ) -> Result<Self, OffsetOverflowError> {
        // Compute the file hash before any normalization.
        let src_hash = SourceFileHash::new_in_memory(hash_kind, src.as_bytes());
        let checksum_hash = checksum_hash_kind.map(|checksum_hash_kind| {
            if checksum_hash_kind == hash_kind {
                src_hash
            } else {
                SourceFileHash::new_in_memory(checksum_hash_kind, src.as_bytes())
            }
        });
        let normalized_pos = normalize_src(&mut src);

        let stable_id = StableSourceFileId::from_filename_in_current_crate(&name);
        let source_len = src.len();
        let source_len = u32::try_from(source_len).map_err(|_| OffsetOverflowError)?;
        if source_len > Self::MAX_FILE_SIZE {
            return Err(OffsetOverflowError);
        }

        let (lines, multibyte_chars) = analyze_source_file::analyze_source_file(&src);

        Ok(SourceFile {
            name,
            src: Some(Arc::new(src)),
            src_hash,
            checksum_hash,
            external_src: FreezeLock::frozen(ExternalSource::Unneeded),
            start_pos: BytePos::from_u32(0),
            source_len: RelativeBytePos::from_u32(source_len),
            lines: FreezeLock::frozen(SourceFileLines::Lines(lines)),
            multibyte_chars,
            normalized_pos,
            stable_id,
            cnum: LOCAL_CRATE,
        })
    }

    /// This converts the `lines` field to contain `SourceFileLines::Lines` if needed and freezes
    /// it.
    fn convert_diffs_to_lines_frozen(&self) {
        let mut guard = if let Some(guard) = self.lines.try_write() { guard } else { return };

        let SourceFileDiffs { bytes_per_diff, num_diffs, raw_diffs } = match &*guard {
            SourceFileLines::Diffs(diffs) => diffs,
            SourceFileLines::Lines(..) => {
                FreezeWriteGuard::freeze(guard);
                return;
            }
        };

        // Convert from "diffs" form to "lines" form.
        let num_lines = num_diffs + 1;
        let mut lines = Vec::with_capacity(num_lines);
        let mut line_start = RelativeBytePos(0);
        lines.push(line_start);

        assert_eq!(*num_diffs, raw_diffs.len() / bytes_per_diff);
        match bytes_per_diff {
            1 => {
                lines.extend(raw_diffs.into_iter().map(|&diff| {
                    line_start = line_start + RelativeBytePos(diff as u32);
                    line_start
                }));
            }
            2 => {
                lines.extend((0..*num_diffs).map(|i| {
                    let pos = bytes_per_diff * i;
                    let bytes = [raw_diffs[pos], raw_diffs[pos + 1]];
                    let diff = u16::from_le_bytes(bytes);
                    line_start = line_start + RelativeBytePos(diff as u32);
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
                    line_start = line_start + RelativeBytePos(diff);
                    line_start
                }));
            }
            _ => unreachable!(),
        }

        *guard = SourceFileLines::Lines(lines);

        FreezeWriteGuard::freeze(guard);
    }

    pub fn lines(&self) -> &[RelativeBytePos] {
        if let Some(SourceFileLines::Lines(lines)) = self.lines.get() {
            return &lines[..];
        }

        outline(|| {
            self.convert_diffs_to_lines_frozen();
            if let Some(SourceFileLines::Lines(lines)) = self.lines.get() {
                return &lines[..];
            }
            unreachable!()
        })
    }

    /// Returns the `BytePos` of the beginning of the current line.
    pub fn line_begin_pos(&self, pos: BytePos) -> BytePos {
        let pos = self.relative_position(pos);
        let line_index = self.lookup_line(pos).unwrap();
        let line_start_pos = self.lines()[line_index];
        self.absolute_position(line_start_pos)
    }

    /// Add externally loaded source.
    /// If the hash of the input doesn't match or no input is supplied via None,
    /// it is interpreted as an error and the corresponding enum variant is set.
    /// The return value signifies whether some kind of source is present.
    pub fn add_external_src<F>(&self, get_src: F) -> bool
    where
        F: FnOnce() -> Option<String>,
    {
        if !self.external_src.is_frozen() {
            let src = get_src();
            let src = src.and_then(|mut src| {
                // The src_hash needs to be computed on the pre-normalized src.
                self.src_hash.matches(&src).then(|| {
                    normalize_src(&mut src);
                    src
                })
            });

            self.external_src.try_write().map(|mut external_src| {
                if let ExternalSource::Foreign {
                    kind: src_kind @ ExternalSourceKind::AbsentOk,
                    ..
                } = &mut *external_src
                {
                    *src_kind = if let Some(src) = src {
                        ExternalSourceKind::Present(Arc::new(src))
                    } else {
                        ExternalSourceKind::AbsentErr
                    };
                } else {
                    panic!("unexpected state {:?}", *external_src)
                }

                // Freeze this so we don't try to load the source again.
                FreezeWriteGuard::freeze(external_src)
            });
        }

        self.src.is_some() || self.external_src.read().get_source().is_some()
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
            let line = self.lines().get(line_number).copied()?;
            line.to_usize()
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
        self.lines().len()
    }

    #[inline]
    pub fn absolute_position(&self, pos: RelativeBytePos) -> BytePos {
        BytePos::from_u32(pos.to_u32() + self.start_pos.to_u32())
    }

    #[inline]
    pub fn relative_position(&self, pos: BytePos) -> RelativeBytePos {
        RelativeBytePos::from_u32(pos.to_u32() - self.start_pos.to_u32())
    }

    #[inline]
    pub fn end_position(&self) -> BytePos {
        self.absolute_position(self.source_len)
    }

    /// Finds the line containing the given position. The return value is the
    /// index into the `lines` array of this `SourceFile`, not the 1-based line
    /// number. If the source_file is empty or the position is located before the
    /// first line, `None` is returned.
    pub fn lookup_line(&self, pos: RelativeBytePos) -> Option<usize> {
        self.lines().partition_point(|x| x <= &pos).checked_sub(1)
    }

    pub fn line_bounds(&self, line_index: usize) -> Range<BytePos> {
        if self.is_empty() {
            return self.start_pos..self.start_pos;
        }

        let lines = self.lines();
        assert!(line_index < lines.len());
        if line_index == (lines.len() - 1) {
            self.absolute_position(lines[line_index])..self.end_position()
        } else {
            self.absolute_position(lines[line_index])..self.absolute_position(lines[line_index + 1])
        }
    }

    /// Returns whether or not the file contains the given `SourceMap` byte
    /// position. The position one past the end of the file is considered to be
    /// contained by the file. This implies that files for which `is_empty`
    /// returns true still contain one byte position according to this function.
    #[inline]
    pub fn contains(&self, byte_pos: BytePos) -> bool {
        byte_pos >= self.start_pos && byte_pos <= self.end_position()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.source_len.to_u32() == 0
    }

    /// Calculates the original byte position relative to the start of the file
    /// based on the given byte position.
    pub fn original_relative_byte_pos(&self, pos: BytePos) -> RelativeBytePos {
        let pos = self.relative_position(pos);

        // Diff before any records is 0. Otherwise use the previously recorded
        // diff as that applies to the following characters until a new diff
        // is recorded.
        let diff = match self.normalized_pos.binary_search_by(|np| np.pos.cmp(&pos)) {
            Ok(i) => self.normalized_pos[i].diff,
            Err(0) => 0,
            Err(i) => self.normalized_pos[i - 1].diff,
        };

        RelativeBytePos::from_u32(pos.0 + diff)
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
            Err(0) => 0,
            Err(i) => self.normalized_pos[i - 1].diff,
        };

        BytePos::from_u32(self.start_pos.0 + offset - diff)
    }

    /// Converts an relative `RelativeBytePos` to a `CharPos` relative to the `SourceFile`.
    fn bytepos_to_file_charpos(&self, bpos: RelativeBytePos) -> CharPos {
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

        assert!(total_extra_bytes <= bpos.to_u32());
        CharPos(bpos.to_usize() - total_extra_bytes as usize)
    }

    /// Looks up the file's (1-based) line number and (0-based `CharPos`) column offset, for a
    /// given `RelativeBytePos`.
    fn lookup_file_pos(&self, pos: RelativeBytePos) -> (usize, CharPos) {
        let chpos = self.bytepos_to_file_charpos(pos);
        match self.lookup_line(pos) {
            Some(a) => {
                let line = a + 1; // Line numbers start at 1
                let linebpos = self.lines()[a];
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
        let pos = self.relative_position(pos);
        let (line, col_or_chpos) = self.lookup_file_pos(pos);
        if line > 0 {
            let Some(code) = self.get_line(line - 1) else {
                // If we don't have the code available, it is ok as a fallback to return the bytepos
                // instead of the "display" column, which is only used to properly show underlines
                // in the terminal.
                // FIXME: we'll want better handling of this in the future for the sake of tools
                // that want to use the display col instead of byte offsets to modify Rust code, but
                // that is a problem for another day, the previous code was already incorrect for
                // both displaying *and* third party tools using the json output naïvely.
                tracing::info!("couldn't find line {line} {:?}", self.name);
                return (line, col_or_chpos, col_or_chpos.0);
            };
            let display_col = code.chars().take(col_or_chpos.0).map(|ch| char_width(ch)).sum();
            (line, col_or_chpos, display_col)
        } else {
            // This is never meant to happen?
            (0, col_or_chpos, col_or_chpos.0)
        }
    }
}

pub fn char_width(ch: char) -> usize {
    // FIXME: `unicode_width` sometimes disagrees with terminals on how wide a `char` is. For now,
    // just accept that sometimes the code line will be longer than desired.
    match ch {
        '\t' => 4,
        // Keep the following list in sync with `rustc_errors::emitter::OUTPUT_REPLACEMENTS`. These
        // are control points that we replace before printing with a visible codepoint for the sake
        // of being able to point at them with underlines.
        '\u{0000}' | '\u{0001}' | '\u{0002}' | '\u{0003}' | '\u{0004}' | '\u{0005}'
        | '\u{0006}' | '\u{0007}' | '\u{0008}' | '\u{000B}' | '\u{000C}' | '\u{000D}'
        | '\u{000E}' | '\u{000F}' | '\u{0010}' | '\u{0011}' | '\u{0012}' | '\u{0013}'
        | '\u{0014}' | '\u{0015}' | '\u{0016}' | '\u{0017}' | '\u{0018}' | '\u{0019}'
        | '\u{001A}' | '\u{001B}' | '\u{001C}' | '\u{001D}' | '\u{001E}' | '\u{001F}'
        | '\u{007F}' | '\u{202A}' | '\u{202B}' | '\u{202D}' | '\u{202E}' | '\u{2066}'
        | '\u{2067}' | '\u{2068}' | '\u{202C}' | '\u{2069}' => 1,
        _ => unicode_width::UnicodeWidthChar::width(ch).unwrap_or(1),
    }
}

pub fn str_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

/// Normalizes the source code and records the normalizations.
fn normalize_src(src: &mut String) -> Vec<NormalizedPos> {
    let mut normalized_pos = vec![];
    remove_bom(src, &mut normalized_pos);
    normalize_newlines(src, &mut normalized_pos);
    normalized_pos
}

/// Removes UTF-8 BOM, if any.
fn remove_bom(src: &mut String, normalized_pos: &mut Vec<NormalizedPos>) {
    if src.starts_with('\u{feff}') {
        src.drain(..3);
        normalized_pos.push(NormalizedPos { pos: RelativeBytePos(0), diff: 3 });
    }
}

/// Replaces `\r\n` with `\n` in-place in `src`.
///
/// Leaves any occurrences of lone `\r` unchanged.
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
            pos: RelativeBytePos::from_usize(cursor + 1),
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

    /// A byte offset relative to file beginning.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
    pub struct RelativeBytePos(pub u32);

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

impl<H: HashStableContext> HashStable<H> for RelativeBytePos {
    fn hash_stable(&self, hcx: &mut H, hasher: &mut StableHasher) {
        self.0.hash_stable(hcx, hasher);
    }
}

impl<S: Encoder> Encodable<S> for RelativeBytePos {
    fn encode(&self, s: &mut S) {
        s.emit_u32(self.0);
    }
}

impl<D: Decoder> Decodable<D> for RelativeBytePos {
    fn decode(d: &mut D) -> RelativeBytePos {
        RelativeBytePos(d.read_u32())
    }
}

// _____________________________________________________________________________
// Loc, SourceFileAndLine, SourceFileAndBytePos
//

/// A source code location used for error reporting.
#[derive(Debug, Clone)]
pub struct Loc {
    /// Information about the original source.
    pub file: Arc<SourceFile>,
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
    pub sf: Arc<SourceFile>,
    /// Index of line, starting from 0.
    pub line: usize,
}
#[derive(Debug)]
pub struct SourceFileAndBytePos {
    pub sf: Arc<SourceFile>,
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
    pub file: Arc<SourceFile>,
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
    ) -> Option<(Arc<SourceFile>, usize, BytePos, usize, BytePos)>;
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
        Hash::hash(&file.stable_id, hasher);

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
///
/// The `()` field is necessary: it is non-`pub`, which means values of this
/// type cannot be constructed outside of this crate.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[derive(HashStable_Generic)]
pub struct ErrorGuaranteed(());

impl ErrorGuaranteed {
    /// Don't use this outside of `DiagCtxtInner::emit_diagnostic`!
    #[deprecated = "should only be used in `DiagCtxtInner::emit_diagnostic`"]
    pub fn unchecked_error_guaranteed() -> Self {
        ErrorGuaranteed(())
    }

    pub fn raise_fatal(self) -> ! {
        FatalError.raise()
    }
}

impl<E: rustc_serialize::Encoder> Encodable<E> for ErrorGuaranteed {
    #[inline]
    fn encode(&self, _e: &mut E) {
        panic!(
            "should never serialize an `ErrorGuaranteed`, as we do not write metadata or \
            incremental caches in case errors occurred"
        )
    }
}
impl<D: rustc_serialize::Decoder> Decodable<D> for ErrorGuaranteed {
    #[inline]
    fn decode(_d: &mut D) -> ErrorGuaranteed {
        panic!(
            "`ErrorGuaranteed` should never have been serialized to metadata or incremental caches"
        )
    }
}
