//! Items common to the prelude of all editions.
//!
//! See the [module-level documentation](super) for more.

// Re-exported core operators
// Re-exported types and traits
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::clone::Clone;
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::cmp::{Eq, Ord, PartialEq, PartialOrd};
#[unstable(
    feature = "concat_bytes",
    issue = "87555",
    reason = "`concat_bytes` is not stable enough for use and is subject to change"
)]
#[doc(no_inline)]
pub use crate::concat_bytes;
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::convert::{AsMut, AsRef, From, Into};
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::default::Default;
// Re-exported built-in macros
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[doc(no_inline)]
pub use crate::fmt::macros::Debug;
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[doc(no_inline)]
pub use crate::hash::macros::Hash;
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::iter::{DoubleEndedIterator, ExactSizeIterator};
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::iter::{Extend, IntoIterator, Iterator};
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
    feature = "deref_patterns",
    issue = "87121",
    reason = "placeholder syntax for deref patterns"
)]
pub use crate::macros::builtin::deref;
#[unstable(feature = "derive_const", issue = "none")]
pub use crate::macros::builtin::derive_const;
#[unstable(
    feature = "type_ascription",
    issue = "23416",
    reason = "placeholder syntax for type ascription"
)]
pub use crate::macros::builtin::type_ascribe;
// Do not `doc(no_inline)` so that they become doc items on their own
// (no public module for them to be re-exported from).
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
pub use crate::macros::builtin::{
    alloc_error_handler, bench, derive, global_allocator, test, test_case,
};
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::marker::{Copy, Send, Sized, Sync, Unpin};
// Re-exported functions
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::mem::drop;
#[stable(feature = "size_of_prelude", since = "1.80.0")]
#[doc(no_inline)]
pub use crate::mem::{align_of, align_of_val, size_of, size_of_val};
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::ops::{Drop, Fn, FnMut, FnOnce};
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::option::Option::{self, None, Some};
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::result::Result::{self, Err, Ok};
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[allow(deprecated)]
#[doc(no_inline)]
pub use crate::{
    assert, cfg, column, compile_error, concat, concat_idents, env, file, format_args,
    format_args_nl, include, include_bytes, include_str, line, log_syntax, module_path, option_env,
    stringify, trace_macros,
};
