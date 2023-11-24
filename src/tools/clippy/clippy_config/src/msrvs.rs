use rustc_ast::Attribute;
use rustc_semver::RustcVersion;
use rustc_session::Session;
use rustc_span::{sym, Symbol};
use serde::Deserialize;

macro_rules! msrv_aliases {
    ($($major:literal,$minor:literal,$patch:literal {
        $($name:ident),* $(,)?
    })*) => {
        $($(
        pub const $name: RustcVersion = RustcVersion::new($major, $minor, $patch);
        )*)*
    };
}

// names may refer to stabilized feature flags or library items
msrv_aliases! {
    1,71,0 { TUPLE_ARRAY_CONVERSIONS, BUILD_HASHER_HASH_ONE }
    1,70,0 { OPTION_IS_SOME_AND, BINARY_HEAP_RETAIN }
    1,68,0 { PATH_MAIN_SEPARATOR_STR }
    1,65,0 { LET_ELSE, POINTER_CAST_CONSTNESS }
    1,62,0 { BOOL_THEN_SOME, DEFAULT_ENUM_ATTRIBUTE }
    1,58,0 { FORMAT_ARGS_CAPTURE, PATTERN_TRAIT_CHAR_ARRAY }
    1,55,0 { SEEK_REWIND }
    1,53,0 { OR_PATTERNS, MANUAL_BITS, BTREE_MAP_RETAIN, BTREE_SET_RETAIN, ARRAY_INTO_ITERATOR }
    1,52,0 { STR_SPLIT_ONCE, REM_EUCLID_CONST }
    1,51,0 { BORROW_AS_PTR, SEEK_FROM_CURRENT, UNSIGNED_ABS }
    1,50,0 { BOOL_THEN, CLAMP }
    1,47,0 { TAU, IS_ASCII_DIGIT_CONST, ARRAY_IMPL_ANY_LEN }
    1,46,0 { CONST_IF_MATCH }
    1,45,0 { STR_STRIP_PREFIX }
    1,43,0 { LOG2_10, LOG10_2 }
    1,42,0 { MATCHES_MACRO, SLICE_PATTERNS, PTR_SLICE_RAW_PARTS }
    1,41,0 { RE_REBALANCING_COHERENCE, RESULT_MAP_OR_ELSE }
    1,40,0 { MEM_TAKE, NON_EXHAUSTIVE, OPTION_AS_DEREF }
    1,38,0 { POINTER_CAST, REM_EUCLID }
    1,37,0 { TYPE_ALIAS_ENUM_VARIANTS }
    1,36,0 { ITERATOR_COPIED }
    1,35,0 { OPTION_COPIED, RANGE_CONTAINS }
    1,34,0 { TRY_FROM }
    1,30,0 { ITERATOR_FIND_MAP, TOOL_ATTRIBUTES }
    1,28,0 { FROM_BOOL }
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
    stack: Vec<RustcVersion>,
}

impl<'de> Deserialize<'de> for Msrv {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = String::deserialize(deserializer)?;
        RustcVersion::parse(&v)
            .map(|v| Msrv { stack: vec![v] })
            .map_err(|_| serde::de::Error::custom("not a valid Rust version"))
    }
}

impl Msrv {
    pub fn empty() -> Msrv {
        Msrv { stack: Vec::new() }
    }

    pub fn read_cargo(&mut self, sess: &Session) {
        let cargo_msrv = std::env::var("CARGO_PKG_RUST_VERSION")
            .ok()
            .and_then(|v| RustcVersion::parse(&v).ok());

        match (self.current(), cargo_msrv) {
            (None, Some(cargo_msrv)) => self.stack = vec![cargo_msrv],
            (Some(clippy_msrv), Some(cargo_msrv)) => {
                if clippy_msrv != cargo_msrv {
                    sess.warn(format!(
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
        self.current().map_or(true, |version| version.meets(required))
    }

    fn parse_attr(sess: &Session, attrs: &[Attribute]) -> Option<RustcVersion> {
        let sym_msrv = Symbol::intern("msrv");
        let mut msrv_attrs = attrs.iter().filter(|attr| attr.path_matches(&[sym::clippy, sym_msrv]));

        if let Some(msrv_attr) = msrv_attrs.next() {
            if let Some(duplicate) = msrv_attrs.last() {
                sess.struct_span_err(duplicate.span, "`clippy::msrv` is defined multiple times")
                    .span_note(msrv_attr.span, "first definition found here")
                    .emit();
            }

            if let Some(msrv) = msrv_attr.value_str() {
                if let Ok(version) = RustcVersion::parse(msrv.as_str()) {
                    return Some(version);
                }

                sess.span_err(msrv_attr.span, format!("`{msrv}` is not a valid Rust version"));
            } else {
                sess.span_err(msrv_attr.span, "bad clippy attribute");
            }
        }

        None
    }

    pub fn enter_lint_attrs(&mut self, sess: &Session, attrs: &[Attribute]) {
        if let Some(version) = Self::parse_attr(sess, attrs) {
            self.stack.push(version);
        }
    }

    pub fn exit_lint_attrs(&mut self, sess: &Session, attrs: &[Attribute]) {
        if Self::parse_attr(sess, attrs).is_some() {
            self.stack.pop();
        }
    }
}
