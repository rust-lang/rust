use crate::{is_in_const_context, sym};
use rustc_ast::Attribute;
use rustc_ast::attr::AttributeExt;
use rustc_attr_parsing::parse_version;
use rustc_data_structures::smallvec::SmallVec;
use rustc_hir::attrs::RustcVersion;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::{Constness, HirId, StabilityLevel, StableSince};
use rustc_lint::LateContext;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use std::iter::once;
use std::sync::atomic::{AtomicBool, Ordering};

macro_rules! msrv_aliases {
    ($($major:literal,$minor:literal,$patch:literal {
        $($name:ident),* $(,)?
    })*) => {
        $($(
        pub const $name: RustcVersion = RustcVersion { major: $major, minor :$minor, patch: $patch };
        )*)*
    };
}

// names may refer to stabilized feature flags or library items
msrv_aliases! {
    1,97,0 { ISOLATE_LOWEST_ONE, BIT_WIDTH }
    1,94,0 { EULER_GAMMA, GOLDEN_RATIO }
    1,93,0 { VEC_DEQUE_POP_BACK_IF, VEC_DEQUE_POP_FRONT_IF }
    1,91,0 { DURATION_FROM_MINUTES_HOURS }
    1,89,0 { NONNULL_FROM_MUT }
    1,88,0 { LET_CHAINS, AS_CHUNKS, RAW_PTR_DEFAULT }
    1,87,0 { OS_STR_DISPLAY, INT_MIDPOINT, CONST_CHAR_IS_DIGIT, UNSIGNED_IS_MULTIPLE_OF, INTEGER_SIGN_CAST }
    1,86,0 { VEC_POP_IF }
    1,85,0 { UINT_FLOAT_MIDPOINT, CONST_SIZE_OF_VAL, WAKER_NOOP }
    1,84,0 { CONST_OPTION_AS_SLICE, MANUAL_DANGLING_PTR }
    1,83,0 { CONST_EXTERN_FN, CONST_FLOAT_BITS_CONV, CONST_FLOAT_CLASSIFY, CONST_MUT_REFS, CONST_UNWRAP }
    1,82,0 { IS_NONE_OR, REPEAT_N, RAW_REF_OP, SPECIALIZED_TO_STRING_FOR_REFS }
    1,81,0 { LINT_REASONS_STABILIZATION, ERROR_IN_CORE, EXPLICIT_SELF_TYPE_ELISION, DURATION_ABS_DIFF }
    1,80,0 { BOX_INTO_ITER, LAZY_CELL }
    1,79,0 { CONST_BLOCKS, CSTR_COUNT_BYTES }
    1,77,0 { C_STR_LITERALS }
    1,76,0 { PTR_FROM_REF, OPTION_RESULT_INSPECT }
    1,75,0 { OPTION_AS_SLICE }
    1,74,0 { REPR_RUST, IO_ERROR_OTHER }
    1,73,0 { DIV_CEIL }
    1,71,0 { TUPLE_ARRAY_CONVERSIONS, BUILD_HASHER_HASH_ONE }
    1,70,0 { OPTION_RESULT_IS_VARIANT_AND, BINARY_HEAP_RETAIN }
    1,68,0 { PATH_MAIN_SEPARATOR_STR }
    1,67,0 { ILOG2 }
    1,65,0 { LET_ELSE, POINTER_CAST_CONSTNESS }
    1,63,0 { CLONE_INTO, CONST_SLICE_FROM_REF }
    1,62,0 { BOOL_THEN_SOME, DEFAULT_ENUM_ATTRIBUTE, CONST_EXTERN_C_FN }
    1,61,0 { CONST_FN_TRAIT_BOUND }
    1,60,0 { ABS_DIFF }
    1,59,0 { THREAD_LOCAL_CONST_INIT }
    1,58,0 { FORMAT_ARGS_CAPTURE, PATTERN_TRAIT_CHAR_ARRAY, CONST_RAW_PTR_DEREF }
    1,57,0 { MAP_WHILE, CONST_PANIC }
    1,56,0 { CONST_FN_UNION }
    1,55,0 { SEEK_REWIND }
    1,54,0 { INTO_KEYS }
    1,53,0 { OR_PATTERNS, INTEGER_BITS, BTREE_MAP_RETAIN, BTREE_SET_RETAIN, ARRAY_INTO_ITERATOR }
    1,52,0 { STR_SPLIT_ONCE, REM_EUCLID_CONST }
    1,51,0 { BORROW_AS_PTR, SEEK_FROM_CURRENT, UNSIGNED_ABS }
    1,50,0 { BOOL_THEN, CLAMP, SLICE_FILL }
    1,47,0 { TAU, IS_ASCII_DIGIT_CONST, ARRAY_IMPL_ANY_LEN, SATURATING_SUB_CONST }
    1,46,0 { CONST_IF_MATCH, OPTION_ZIP }
    1,45,0 { STR_STRIP_PREFIX }
    1,43,0 { LOG2_10, LOG10_2, NUMERIC_ASSOCIATED_CONSTANTS }
    1,42,0 { MATCHES_MACRO, SLICE_PATTERNS, PTR_SLICE_RAW_PARTS }
    1,41,0 { RE_REBALANCING_COHERENCE, RESULT_MAP_OR, RESULT_MAP_OR_ELSE }
    1,40,0 { MEM_TAKE, NON_EXHAUSTIVE, OPTION_AS_DEREF }
    1,38,0 { POINTER_CAST, REM_EUCLID }
    1,37,0 { TYPE_ALIAS_ENUM_VARIANTS }
    1,36,0 { ITERATOR_COPIED }
    1,35,0 { OPTION_COPIED, RANGE_CONTAINS }
    1,34,0 { TRY_FROM }
    1,33,0 { UNDERSCORE_IMPORTS }
    1,32,0 { CONST_IS_POWER_OF_TWO, CONST_DURATION_FROM_NANOS_MICROS_MILLIS_SECS }
    1,31,0 { OPTION_REPLACE }
    1,30,0 { ITERATOR_FIND_MAP, TOOL_ATTRIBUTES }
    1,29,0 { ITER_FLATTEN }
    1,28,0 { FROM_BOOL, REPEAT_WITH, SLICE_FROM_REF }
    1,27,0 { ITERATOR_TRY_FOLD, DOUBLE_ENDED_ITERATOR_RFIND, DURATION_FROM_NANOS_MICROS }
    1,26,0 { RANGE_INCLUSIVE, STRING_RETAIN, POINTER_ADD_SUB_METHODS, BOX_LEAK }
    1,24,0 { IS_ASCII_DIGIT, PTR_NULL }
    1,18,0 { HASH_MAP_RETAIN, HASH_SET_RETAIN }
    1,17,0 { FIELD_INIT_SHORTHAND, STATIC_IN_CONST, EXPECT_ERR }
    1,16,0 { STR_REPEAT, RESULT_UNWRAP_OR_DEFAULT }
    1,15,0 { MAYBE_BOUND_IN_WHERE }
    1,13,0 { QUESTION_MARK_OPERATOR }
    1,3,0 { DURATION_FROM_MILLIS_SECS }
}

/// `#[clippy::msrv]` attributes are rarely used outside of Clippy's test suite, as a basic
/// optimization we can skip traversing the HIR in [`Msrv::meets`] if we never saw an MSRV attribute
/// during the early lint passes
static SEEN_MSRV_ATTR: AtomicBool = AtomicBool::new(false);

/// Tracks the current MSRV from `clippy.toml`, `Cargo.toml` or set via `#[clippy::msrv]` in late
/// lint passes, use [`MsrvStack`] for early passes
#[derive(Copy, Clone, Debug, Default)]
pub struct Msrv(Option<RustcVersion>);

impl From<Option<RustcVersion>> for Msrv {
    #[inline]
    fn from(value: Option<RustcVersion>) -> Self {
        Self(value)
    }
}

impl Msrv {
    /// Returns the MSRV at the current node
    ///
    /// If the crate being linted uses an `#[clippy::msrv]` attribute this will search the parent
    /// nodes for that attribute, prefer to run this check after cheaper pattern matching operations
    pub fn current(self, cx: &LateContext<'_>) -> Option<RustcVersion> {
        if SEEN_MSRV_ATTR.load(Ordering::Relaxed) {
            self.for_attrs(cx.tcx, cx.last_node_with_lint_attrs)
        } else {
            self.0
        }
    }

    /// Returns the MSRV at the specified node
    ///
    /// If the crate being linted uses an `#[clippy::msrv]` attribute this will search the parent
    /// nodes for that attribute, prefer to run this check after cheaper pattern matching operations
    pub fn at(self, tcx: TyCtxt<'_>, node: HirId) -> Option<RustcVersion> {
        if SEEN_MSRV_ATTR.load(Ordering::Relaxed) {
            self.for_attrs(tcx, node)
        } else {
            self.0
        }
    }

    fn for_attrs(self, tcx: TyCtxt<'_>, node: HirId) -> Option<RustcVersion> {
        once(node)
            .chain(tcx.hir_parent_id_iter(node))
            .find_map(|id| parse_attrs(tcx.hir_attrs(id)))
            .or(self.0)
    }

    /// Checks if a required version from [this module](self) is met at the current node
    ///
    /// If the crate being linted uses an `#[clippy::msrv]` attribute this will search the parent
    /// nodes for that attribute, prefer to run this check after cheaper pattern matching operations
    pub fn meets(self, cx: &LateContext<'_>, required: RustcVersion) -> bool {
        self.current(cx).is_none_or(|msrv| msrv >= required)
    }
    /// Checks if a required version from [this module](self) is met at the specified node
    ///
    /// If the crate being linted uses an `#[clippy::msrv]` attribute this will search the parent
    /// nodes for that attribute, prefer to run this check after cheaper pattern matching operations
    pub fn meets_at(self, tcx: TyCtxt<'_>, node: HirId, required: RustcVersion) -> bool {
        self.at(tcx, node).is_none_or(|msrv| msrv >= required)
    }

    pub fn is_stable(self, cx: &LateContext<'_>, def_id: DefId) -> bool {
        self.stability_met(cx, cx.tcx.lookup_stability(def_id).map(|stability| stability.level))
    }

    /// Checks whether `def_id` is `const` and const-stable since a version met by the MSRV.
    ///
    /// `def_id` must identify a function-like definition or an impl.
    ///
    /// Nothing in the crate being linted carries a const-stability attribute, so `const` fns and
    /// impls defined there are treated as meeting any MSRV, mirroring
    /// [`is_stable`](Self::is_stable).
    pub fn is_const_stable(self, cx: &LateContext<'_>, def_id: DefId) -> bool {
        let constness = match cx.tcx.def_kind(def_id) {
            // The constness of a trait impl is not encoded in crate metadata, where `constness`
            // would decode as its default of `Const`. It is only available from the impl header.
            DefKind::Impl { of_trait: true } => cx.tcx.impl_trait_header(def_id).constness,
            _ => cx.tcx.constness(def_id),
        };

        matches!(constness, Constness::Const { .. })
            && self.stability_met(
                cx,
                cx.tcx.lookup_const_stability(def_id).map(|stability| stability.level),
            )
    }

    /// Checks the stability relevant to where we are: const-stability inside a `const` context,
    /// regular stability everywhere else.
    ///
    /// Like [`is_in_const_context`], this requires the `LateContext` to have an enclosing body.
    pub fn is_stable_or_const_stable(self, cx: &LateContext<'_>, def_id: DefId) -> bool {
        if is_in_const_context(cx) {
            self.is_const_stable(cx, def_id)
        } else {
            self.is_stable(cx, def_id)
        }
    }

    fn stability_met(self, cx: &LateContext<'_>, level: Option<StabilityLevel>) -> bool {
        level.is_none_or(|level| {
            if let StabilityLevel::Stable { since, .. } = level {
                let version = match since {
                    StableSince::Version(version) => version,
                    StableSince::Current => RustcVersion::CURRENT,
                    StableSince::Err(_) => return false,
                };

                self.meets(cx, version)
            } else {
                // Unstable fn.
                // FIXME: can we check that the feature is enabled?
                // Please see https://github.com/rust-lang/rust-clippy/pull/17309#discussion_r3486693263 for false-positive concerns.
                false
            }
        })
    }
}

/// Tracks the current MSRV from `clippy.toml`, `Cargo.toml` or set via `#[clippy::msrv]` in early
/// lint passes, use [`Msrv`] for late passes
#[derive(Debug, Clone)]
pub struct MsrvStack {
    stack: SmallVec<[RustcVersion; 2]>,
}

impl From<Option<RustcVersion>> for MsrvStack {
    #[inline]
    fn from(value: Option<RustcVersion>) -> Self {
        Self {
            stack: SmallVec::from_iter(value),
        }
    }
}

impl MsrvStack {
    pub fn current(&self) -> Option<RustcVersion> {
        self.stack.last().copied()
    }

    pub fn meets(&self, required: RustcVersion) -> bool {
        self.current().is_none_or(|msrv| msrv >= required)
    }

    pub fn check_attributes(&mut self, attrs: &[Attribute]) {
        if let Some(version) = parse_attrs(attrs) {
            SEEN_MSRV_ATTR.store(true, Ordering::Relaxed);
            self.stack.push(version);
        }
    }

    pub fn check_attributes_post(&mut self, attrs: &[Attribute]) {
        if parse_attrs(attrs).is_some() {
            self.stack.pop();
        }
    }
}

fn parse_attrs(attrs: &[impl AttributeExt]) -> Option<RustcVersion> {
    let msrv_attr = attrs.iter().find(|attr| attr.path_matches(&[sym::clippy, sym::msrv]))?;

    let msrv = msrv_attr.value_str()?;

    parse_version(msrv)
}

pub fn check_attrs(sess: &Session, attrs: &[impl AttributeExt]) {
    let mut msrv_attrs = attrs.iter().filter(|attr| attr.path_matches(&[sym::clippy, sym::msrv]));

    let Some(msrv_attr) = msrv_attrs.next() else {
        return;
    };

    if let Some(duplicate) = msrv_attrs.next_back() {
        sess.dcx()
            .struct_span_err(duplicate.span(), "`clippy::msrv` is defined multiple times")
            .with_span_note(msrv_attr.span(), "first definition found here")
            .emit();
    }

    let Some(msrv) = msrv_attr.value_str() else {
        sess.dcx().span_err(msrv_attr.span(), "bad clippy attribute");
        return;
    };

    if parse_version(msrv).is_none() {
        sess.dcx()
            .span_err(msrv_attr.span(), format!("`{msrv}` is not a valid Rust version"));
    }
}
