use crate::sym;
use rustc_ast::Attribute;
use rustc_ast::attr::AttributeExt;
use rustc_attr_parsing::{RustcVersion, parse_version};
use rustc_lint::LateContext;
use rustc_session::Session;
use rustc_span::Symbol;
use serde::Deserialize;
use smallvec::SmallVec;
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
    1,88,0 { LET_CHAINS }
    1,87,0 { OS_STR_DISPLAY, INT_MIDPOINT, CONST_CHAR_IS_DIGIT }
    1,85,0 { UINT_FLOAT_MIDPOINT }
    1,84,0 { CONST_OPTION_AS_SLICE, MANUAL_DANGLING_PTR }
    1,83,0 { CONST_EXTERN_FN, CONST_FLOAT_BITS_CONV, CONST_FLOAT_CLASSIFY, CONST_MUT_REFS, CONST_UNWRAP }
    1,82,0 { IS_NONE_OR, REPEAT_N, RAW_REF_OP }
    1,81,0 { LINT_REASONS_STABILIZATION, ERROR_IN_CORE, EXPLICIT_SELF_TYPE_ELISION, DURATION_ABS_DIFF }
    1,80,0 { BOX_INTO_ITER, LAZY_CELL }
    1,77,0 { C_STR_LITERALS }
    1,76,0 { PTR_FROM_REF, OPTION_RESULT_INSPECT }
    1,75,0 { OPTION_AS_SLICE }
    1,74,0 { REPR_RUST, IO_ERROR_OTHER }
    1,73,0 { DIV_CEIL }
    1,71,0 { TUPLE_ARRAY_CONVERSIONS, BUILD_HASHER_HASH_ONE }
    1,70,0 { OPTION_RESULT_IS_VARIANT_AND, BINARY_HEAP_RETAIN }
    1,68,0 { PATH_MAIN_SEPARATOR_STR }
    1,65,0 { LET_ELSE, POINTER_CAST_CONSTNESS }
    1,63,0 { CLONE_INTO, CONST_SLICE_FROM_REF }
    1,62,0 { BOOL_THEN_SOME, DEFAULT_ENUM_ATTRIBUTE, CONST_EXTERN_C_FN }
    1,60,0 { ABS_DIFF }
    1,59,0 { THREAD_LOCAL_CONST_INIT }
    1,58,0 { FORMAT_ARGS_CAPTURE, PATTERN_TRAIT_CHAR_ARRAY, CONST_RAW_PTR_DEREF }
    1,57,0 { MAP_WHILE }
    1,56,0 { CONST_FN_UNION }
    1,55,0 { SEEK_REWIND }
    1,54,0 { INTO_KEYS }
    1,53,0 { OR_PATTERNS, INTEGER_BITS, BTREE_MAP_RETAIN, BTREE_SET_RETAIN, ARRAY_INTO_ITERATOR }
    1,52,0 { STR_SPLIT_ONCE, REM_EUCLID_CONST }
    1,51,0 { BORROW_AS_PTR, SEEK_FROM_CURRENT, UNSIGNED_ABS }
    1,50,0 { BOOL_THEN, CLAMP, SLICE_FILL }
    1,47,0 { TAU, IS_ASCII_DIGIT_CONST, ARRAY_IMPL_ANY_LEN, SATURATING_SUB_CONST }
    1,46,0 { CONST_IF_MATCH }
    1,45,0 { STR_STRIP_PREFIX }
    1,43,0 { LOG2_10, LOG10_2, NUMERIC_ASSOCIATED_CONSTANTS }
    1,42,0 { MATCHES_MACRO, SLICE_PATTERNS, PTR_SLICE_RAW_PARTS }
    1,41,0 { RE_REBALANCING_COHERENCE, RESULT_MAP_OR_ELSE }
    1,40,0 { MEM_TAKE, NON_EXHAUSTIVE, OPTION_AS_DEREF }
    1,38,0 { POINTER_CAST, REM_EUCLID }
    1,37,0 { TYPE_ALIAS_ENUM_VARIANTS }
    1,36,0 { ITERATOR_COPIED }
    1,35,0 { OPTION_COPIED, RANGE_CONTAINS }
    1,34,0 { TRY_FROM }
    1,33,0 { UNDERSCORE_IMPORTS }
    1,32,0 { CONST_IS_POWER_OF_TWO }
    1,31,0 { OPTION_REPLACE }
    1,30,0 { ITERATOR_FIND_MAP, TOOL_ATTRIBUTES }
    1,29,0 { ITER_FLATTEN }
    1,28,0 { FROM_BOOL, REPEAT_WITH, SLICE_FROM_REF }
    1,27,0 { ITERATOR_TRY_FOLD }
    1,26,0 { RANGE_INCLUSIVE, STRING_RETAIN }
    1,24,0 { IS_ASCII_DIGIT }
    1,18,0 { HASH_MAP_RETAIN, HASH_SET_RETAIN }
    1,17,0 { FIELD_INIT_SHORTHAND, STATIC_IN_CONST, EXPECT_ERR }
    1,16,0 { STR_REPEAT }
    1,15,0 { MAYBE_BOUND_IN_WHERE }
    1,13,0 { QUESTION_MARK_OPERATOR }
}

/// `#[clippy::msrv]` attributes are rarely used outside of Clippy's test suite, as a basic
/// optimization we can skip traversing the HIR in [`Msrv::meets`] if we never saw an MSRV attribute
/// during the early lint passes
static SEEN_MSRV_ATTR: AtomicBool = AtomicBool::new(false);

/// Tracks the current MSRV from `clippy.toml`, `Cargo.toml` or set via `#[clippy::msrv]` in late
/// lint passes, use [`MsrvStack`] for early passes
#[derive(Copy, Clone, Debug, Default)]
pub struct Msrv(Option<RustcVersion>);

impl<'de> Deserialize<'de> for Msrv {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = String::deserialize(deserializer)?;
        parse_version(Symbol::intern(&v))
            .map(|v| Self(Some(v)))
            .ok_or_else(|| serde::de::Error::custom("not a valid Rust version"))
    }
}

impl Msrv {
    /// Returns the MSRV at the current node
    ///
    /// If the crate being linted uses an `#[clippy::msrv]` attribute this will search the parent
    /// nodes for that attribute, prefer to run this check after cheaper pattern matching operations
    pub fn current(self, cx: &LateContext<'_>) -> Option<RustcVersion> {
        if SEEN_MSRV_ATTR.load(Ordering::Relaxed) {
            let start = cx.last_node_with_lint_attrs;
            if let Some(msrv_attr) = once(start)
                .chain(cx.tcx.hir_parent_id_iter(start))
                .find_map(|id| parse_attrs(cx.tcx.sess, cx.tcx.hir_attrs(id)))
            {
                return Some(msrv_attr);
            }
        }

        self.0
    }

    /// Checks if a required version from [this module](self) is met at the current node
    ///
    /// If the crate being linted uses an `#[clippy::msrv]` attribute this will search the parent
    /// nodes for that attribute, prefer to run this check after cheaper pattern matching operations
    pub fn meets(self, cx: &LateContext<'_>, required: RustcVersion) -> bool {
        self.current(cx).is_none_or(|msrv| msrv >= required)
    }

    pub fn read_cargo(&mut self, sess: &Session) {
        let cargo_msrv = std::env::var("CARGO_PKG_RUST_VERSION")
            .ok()
            .and_then(|v| parse_version(Symbol::intern(&v)));

        match (self.0, cargo_msrv) {
            (None, Some(cargo_msrv)) => self.0 = Some(cargo_msrv),
            (Some(clippy_msrv), Some(cargo_msrv)) => {
                if clippy_msrv != cargo_msrv {
                    sess.dcx().warn(format!(
                        "the MSRV in `clippy.toml` and `Cargo.toml` differ; using `{clippy_msrv}` from `clippy.toml`"
                    ));
                }
            },
            _ => {},
        }
    }
}

/// Tracks the current MSRV from `clippy.toml`, `Cargo.toml` or set via `#[clippy::msrv]` in early
/// lint passes, use [`Msrv`] for late passes
#[derive(Debug, Clone)]
pub struct MsrvStack {
    stack: SmallVec<[RustcVersion; 2]>,
}

impl MsrvStack {
    pub fn new(initial: Msrv) -> Self {
        Self {
            stack: SmallVec::from_iter(initial.0),
        }
    }

    pub fn current(&self) -> Option<RustcVersion> {
        self.stack.last().copied()
    }

    pub fn meets(&self, required: RustcVersion) -> bool {
        self.current().is_none_or(|msrv| msrv >= required)
    }

    pub fn check_attributes(&mut self, sess: &Session, attrs: &[Attribute]) {
        if let Some(version) = parse_attrs(sess, attrs) {
            SEEN_MSRV_ATTR.store(true, Ordering::Relaxed);
            self.stack.push(version);
        }
    }

    pub fn check_attributes_post(&mut self, sess: &Session, attrs: &[Attribute]) {
        if parse_attrs(sess, attrs).is_some() {
            self.stack.pop();
        }
    }
}

fn parse_attrs(sess: &Session, attrs: &[impl AttributeExt]) -> Option<RustcVersion> {
    let mut msrv_attrs = attrs.iter().filter(|attr| attr.path_matches(&[sym::clippy, sym::msrv]));

    if let Some(msrv_attr) = msrv_attrs.next() {
        if let Some(duplicate) = msrv_attrs.next_back() {
            sess.dcx()
                .struct_span_err(duplicate.span(), "`clippy::msrv` is defined multiple times")
                .with_span_note(msrv_attr.span(), "first definition found here")
                .emit();
        }

        if let Some(msrv) = msrv_attr.value_str() {
            if let Some(version) = parse_version(msrv) {
                return Some(version);
            }

            sess.dcx()
                .span_err(msrv_attr.span(), format!("`{msrv}` is not a valid Rust version"));
        } else {
            sess.dcx().span_err(msrv_attr.span(), "bad clippy attribute");
        }
    }

    None
}
