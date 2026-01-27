//! The first version of the core prelude.
//!
//! See the [module-level documentation](super) for more.

#![stable(feature = "core_prelude", since = "1.4.0")]

// No formatting: this file is nothing but re-exports, and their order is worth preserving.
#![cfg_attr(rustfmt, rustfmt::skip)]

// Re-exported core operators
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::marker::{Copy, Send, Sized, Sync, Unpin};
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::ops::{Drop, Fn, FnMut, FnOnce};
#[stable(feature = "async_closure", since = "1.85.0")]
#[doc(no_inline)]
pub use crate::ops::{AsyncFn, AsyncFnMut, AsyncFnOnce};

// Re-exported functions
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::mem::drop;
#[stable(feature = "size_of_prelude", since = "1.80.0")]
#[doc(no_inline)]
pub use crate::mem::{align_of, align_of_val, size_of, size_of_val};

// Re-exported types and traits
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::clone::Clone;
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::cmp::{Eq, Ord, PartialEq, PartialOrd};
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::convert::{AsMut, AsRef, From, Into};
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::default::Default;
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::iter::{DoubleEndedIterator, ExactSizeIterator, Extend, IntoIterator, Iterator};
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::option::Option::{self, None, Some};
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::result::Result::{self, Err, Ok};

// Re-exported built-in macros
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[doc(no_inline)]
pub use crate::fmt::macros::Debug;
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[doc(no_inline)]
pub use crate::hash::macros::Hash;

#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[doc(no_inline)]
#[expect(deprecated)]
pub use crate::{
    assert, assert_eq, assert_ne, cfg, column, compile_error, concat, debug_assert, debug_assert_eq,
    debug_assert_ne, file, format_args, include, include_bytes, include_str, line, matches,
    module_path, option_env, stringify, todo, r#try, unimplemented, unreachable, write, writeln,
};

// These macros need special handling, so that we don't export them *and* the modules of the same
// name. We only want the macros in the prelude so we shadow the original modules with private
// modules with the same names.
mod ambiguous_macros_only {
    mod env {}
    #[expect(hidden_glob_reexports)]
    mod panic {}
    #[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
    pub use crate::*;
}
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[doc(no_inline)]
pub use self::ambiguous_macros_only::{env, panic};

#[unstable(feature = "cfg_select", issue = "115585")]
#[doc(no_inline)]
pub use crate::cfg_select;

#[unstable(
    feature = "concat_bytes",
    issue = "87555",
    reason = "`concat_bytes` is not stable enough for use and is subject to change"
)]
#[doc(no_inline)]
pub use crate::concat_bytes;

#[unstable(feature = "const_format_args", issue = "none")]
#[doc(no_inline)]
pub use crate::const_format_args;

#[unstable(
    feature = "log_syntax",
    issue = "29598",
    reason = "`log_syntax!` is not stable enough for use and is subject to change"
)]
#[doc(no_inline)]
pub use crate::log_syntax;

#[unstable(feature = "pattern_type_macro", issue = "123646")]
#[doc(no_inline)]
pub use crate::pattern_type;

#[unstable(
    feature = "trace_macros",
    issue = "29598",
    reason = "`trace_macros` is not stable enough for use and is subject to change"
)]
#[doc(no_inline)]
pub use crate::trace_macros;

// Do not `doc(no_inline)` so that they become doc items on their own
// (no public module for them to be re-exported from).
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
pub use crate::macros::builtin::{
    alloc_error_handler, bench, derive, global_allocator, test, test_case,
};

#[unstable(feature = "derive_const", issue = "118304")]
pub use crate::macros::builtin::derive_const;

#[unstable(
    feature = "cfg_accessible",
    issue = "64797",
    reason = "`cfg_accessible` is not fully implemented"
)]
pub use crate::macros::builtin::cfg_accessible;

#[unstable(
    feature = "cfg_eval",
    issue = "82679",
    reason = "`cfg_eval` is a recently implemented feature"
)]
pub use crate::macros::builtin::cfg_eval;

#[unstable(
    feature = "type_ascription",
    issue = "23416",
    reason = "placeholder syntax for type ascription"
)]
pub use crate::macros::builtin::type_ascribe;

#[unstable(
    feature = "deref_patterns",
    issue = "87121",
    reason = "placeholder syntax for deref patterns"
)]
pub use crate::macros::builtin::deref;

#[unstable(
    feature = "type_alias_impl_trait",
    issue = "63063",
    reason = "`type_alias_impl_trait` has open design concerns"
)]
pub use crate::macros::builtin::define_opaque;

#[unstable(feature = "extern_item_impls", issue = "125418")]
pub use crate::macros::builtin::{eii, unsafe_eii};

#[unstable(feature = "eii_internals", issue = "none")]
pub use crate::macros::builtin::eii_declaration;
