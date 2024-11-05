use rustc_ast::attr::AttributeExt;
use rustc_attr::parse_version;
use rustc_hir::RustcVersion;
use rustc_session::Session;
use rustc_span::{Symbol, sym};
use serde::Deserialize;
use smallvec::{SmallVec, smallvec};
use std::fmt;

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
    1,83,0 { CONST_EXTERN_FN, CONST_FLOAT_BITS_CONV, CONST_FLOAT_CLASSIFY }
    1,82,0 { IS_NONE_OR, REPEAT_N }
    1,81,0 { LINT_REASONS_STABILIZATION }
    1,80,0 { BOX_INTO_ITER}
    1,77,0 { C_STR_LITERALS }
    1,76,0 { PTR_FROM_REF, OPTION_RESULT_INSPECT }
    1,73,0 { MANUAL_DIV_CEIL }
    1,71,0 { TUPLE_ARRAY_CONVERSIONS, BUILD_HASHER_HASH_ONE }
    1,70,0 { OPTION_RESULT_IS_VARIANT_AND, BINARY_HEAP_RETAIN }
    1,68,0 { PATH_MAIN_SEPARATOR_STR }
    1,65,0 { LET_ELSE, POINTER_CAST_CONSTNESS }
    1,63,0 { CLONE_INTO }
    1,62,0 { BOOL_THEN_SOME, DEFAULT_ENUM_ATTRIBUTE, CONST_EXTERN_C_FN }
    1,59,0 { THREAD_LOCAL_CONST_INIT }
    1,58,0 { FORMAT_ARGS_CAPTURE, PATTERN_TRAIT_CHAR_ARRAY, CONST_RAW_PTR_DEREF }
    1,56,0 { CONST_FN_UNION }
    1,55,0 { SEEK_REWIND }
    1,54,0 { INTO_KEYS }
    1,53,0 { OR_PATTERNS, MANUAL_BITS, BTREE_MAP_RETAIN, BTREE_SET_RETAIN, ARRAY_INTO_ITERATOR }
    1,52,0 { STR_SPLIT_ONCE, REM_EUCLID_CONST }
    1,51,0 { BORROW_AS_PTR, SEEK_FROM_CURRENT, UNSIGNED_ABS }
    1,50,0 { BOOL_THEN, CLAMP }
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
    1,30,0 { ITERATOR_FIND_MAP, TOOL_ATTRIBUTES }
    1,29,0 { ITER_FLATTEN }
    1,28,0 { FROM_BOOL, REPEAT_WITH }
    1,27,0 { ITERATOR_TRY_FOLD }
    1,26,0 { RANGE_INCLUSIVE, STRING_RETAIN }
    1,24,0 { IS_ASCII_DIGIT }
    1,18,0 { HASH_MAP_RETAIN, HASH_SET_RETAIN }
    1,17,0 { FIELD_INIT_SHORTHAND, STATIC_IN_CONST, EXPECT_ERR }
    1,16,0 { STR_REPEAT }
    1,15,0 { MAYBE_BOUND_IN_WHERE }
}

/// Tracks the current MSRV from `clippy.toml`, `Cargo.toml` or set via `#[clippy::msrv]`
#[derive(Debug, Clone)]
pub struct Msrv {
    stack: SmallVec<[RustcVersion; 2]>,
}

impl fmt::Display for Msrv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(msrv) = self.current() {
            write!(f, "{msrv}")
        } else {
            f.write_str("1.0.0")
        }
    }
}

impl<'de> Deserialize<'de> for Msrv {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = String::deserialize(deserializer)?;
        parse_version(Symbol::intern(&v))
            .map(|v| Msrv { stack: smallvec![v] })
            .ok_or_else(|| serde::de::Error::custom("not a valid Rust version"))
    }
}

impl Msrv {
    pub fn empty() -> Msrv {
        Msrv { stack: SmallVec::new() }
    }

    pub fn read_cargo(&mut self, sess: &Session) {
        let cargo_msrv = std::env::var("CARGO_PKG_RUST_VERSION")
            .ok()
            .and_then(|v| parse_version(Symbol::intern(&v)));

        match (self.current(), cargo_msrv) {
            (None, Some(cargo_msrv)) => self.stack = smallvec![cargo_msrv],
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

    pub fn current(&self) -> Option<RustcVersion> {
        self.stack.last().copied()
    }

    pub fn meets(&self, required: RustcVersion) -> bool {
        self.current().is_none_or(|msrv| msrv >= required)
    }

    fn parse_attr(sess: &Session, attrs: &[impl AttributeExt]) -> Option<RustcVersion> {
        let sym_msrv = Symbol::intern("msrv");
        let mut msrv_attrs = attrs.iter().filter(|attr| attr.path_matches(&[sym::clippy, sym_msrv]));

        if let Some(msrv_attr) = msrv_attrs.next() {
            if let Some(duplicate) = msrv_attrs.last() {
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

    pub fn check_attributes(&mut self, sess: &Session, attrs: &[impl AttributeExt]) {
        if let Some(version) = Self::parse_attr(sess, attrs) {
            self.stack.push(version);
        }
    }

    pub fn check_attributes_post(&mut self, sess: &Session, attrs: &[impl AttributeExt]) {
        if Self::parse_attr(sess, attrs).is_some() {
            self.stack.pop();
        }
    }
}
