use std::sync::OnceLock;

use rustc_ast::Attribute;
use rustc_semver::RustcVersion;
use rustc_session::Session;
use rustc_span::Span;

use crate::attrs::get_unique_attr;

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
    1,65,0 { LET_ELSE }
    1,62,0 { BOOL_THEN_SOME }
    1,58,0 { FORMAT_ARGS_CAPTURE }
    1,53,0 { OR_PATTERNS, MANUAL_BITS, BTREE_MAP_RETAIN, BTREE_SET_RETAIN, ARRAY_INTO_ITERATOR }
    1,52,0 { STR_SPLIT_ONCE, REM_EUCLID_CONST }
    1,51,0 { BORROW_AS_PTR, SEEK_FROM_CURRENT, UNSIGNED_ABS }
    1,50,0 { BOOL_THEN, CLAMP }
    1,47,0 { TAU, IS_ASCII_DIGIT_CONST }
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
    1,26,0 { RANGE_INCLUSIVE, STRING_RETAIN }
    1,24,0 { IS_ASCII_DIGIT }
    1,18,0 { HASH_MAP_RETAIN, HASH_SET_RETAIN }
    1,17,0 { FIELD_INIT_SHORTHAND, STATIC_IN_CONST, EXPECT_ERR }
    1,16,0 { STR_REPEAT }
    1,55,0 { SEEK_REWIND }
}

fn parse_msrv(msrv: &str, sess: Option<&Session>, span: Option<Span>) -> Option<RustcVersion> {
    if let Ok(version) = RustcVersion::parse(msrv) {
        return Some(version);
    } else if let Some(sess) = sess {
        if let Some(span) = span {
            sess.span_err(span, format!("`{msrv}` is not a valid Rust version"));
        }
    }
    None
}

/// Tracks the current MSRV from `clippy.toml`, `Cargo.toml` or set via `#[clippy::msrv]`
#[derive(Debug, Clone, Default)]
pub struct Msrv {
    stack: Vec<RustcVersion>,
}

impl Msrv {
    fn new(initial: Option<RustcVersion>) -> Self {
        Self {
            stack: Vec::from_iter(initial),
        }
    }

    fn read_inner(conf_msrv: &Option<String>, sess: &Session) -> Self {
        let cargo_msrv = std::env::var("CARGO_PKG_RUST_VERSION")
            .ok()
            .and_then(|v| parse_msrv(&v, None, None));
        let clippy_msrv = conf_msrv.as_ref().and_then(|s| {
            parse_msrv(s, None, None).or_else(|| {
                sess.err(format!(
                    "error reading Clippy's configuration file. `{s}` is not a valid Rust version"
                ));
                None
            })
        });

        // if both files have an msrv, let's compare them and emit a warning if they differ
        if let Some(cargo_msrv) = cargo_msrv
            && let Some(clippy_msrv) = clippy_msrv
            && clippy_msrv != cargo_msrv
        {
            sess.warn(format!(
                "the MSRV in `clippy.toml` and `Cargo.toml` differ; using `{clippy_msrv}` from `clippy.toml`"
            ));
        }

        Self::new(clippy_msrv.or(cargo_msrv))
    }

    /// Set the initial MSRV from the Clippy config file or from Cargo due to the `rust-version`
    /// field in `Cargo.toml`
    ///
    /// Returns a `&'static Msrv` as `Copy` types are more easily passed to the
    /// `register_{late,early}_pass` callbacks
    pub fn read(conf_msrv: &Option<String>, sess: &Session) -> &'static Self {
        static PARSED: OnceLock<Msrv> = OnceLock::new();

        PARSED.get_or_init(|| Self::read_inner(conf_msrv, sess))
    }

    pub fn current(&self) -> Option<RustcVersion> {
        self.stack.last().copied()
    }

    pub fn meets(&self, required: RustcVersion) -> bool {
        self.current().map_or(true, |version| version.meets(required))
    }

    fn parse_attr(sess: &Session, attrs: &[Attribute]) -> Option<RustcVersion> {
        if let Some(msrv_attr) = get_unique_attr(sess, attrs, "msrv") {
            if let Some(msrv) = msrv_attr.value_str() {
                return parse_msrv(&msrv.to_string(), Some(sess), Some(msrv_attr.span));
            }

            sess.span_err(msrv_attr.span, "bad clippy attribute");
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
