//! The first version of the core prelude.
//!
//! See the [module-level documentation](super) for more.

#![stable(feature = "core_prelude", since = "1.4.0")]

// Re-exported core operators
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::marker::{Copy, Send, Sized, Sync, Unpin};
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::ops::{Drop, Fn, FnMut, FnOnce};

// Re-exported functions
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::mem::drop;

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
pub use crate::iter::{DoubleEndedIterator, ExactSizeIterator};
#[stable(feature = "core_prelude", since = "1.4.0")]
#[doc(no_inline)]
pub use crate::iter::{Extend, IntoIterator, Iterator};
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
#[allow(deprecated)]
#[doc(no_inline)]
pub use crate::{
    assert, cfg, column, compile_error, concat, concat_idents, env, file, format_args,
    format_args_nl, include, include_bytes, include_str, line, llvm_asm, log_syntax, module_path,
    option_env, stringify, trace_macros,
};

#[unstable(
    feature = "asm",
    issue = "72016",
    reason = "inline assembly is not stable enough for use and is subject to change"
)]
#[doc(no_inline)]
pub use crate::arch::asm;

#[unstable(
    feature = "global_asm",
    issue = "35119",
    reason = "`global_asm!` is not stable enough for use and is subject to change"
)]
#[doc(no_inline)]
pub use crate::arch::global_asm;

#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[allow(deprecated, deprecated_in_future)]
#[doc(no_inline)]
pub use crate::macros::builtin::{
    bench, global_allocator, test, test_case, RustcDecodable, RustcEncodable,
};

#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[doc(no_inline)]
pub use crate::macros::builtin::derive;

#[unstable(
    feature = "cfg_accessible",
    issue = "64797",
    reason = "`cfg_accessible` is not fully implemented"
)]
#[doc(no_inline)]
pub use crate::macros::builtin::cfg_accessible;

#[unstable(
    feature = "cfg_eval",
    issue = "82679",
    reason = "`cfg_eval` is a recently implemented feature"
)]
#[doc(no_inline)]
pub use crate::macros::builtin::cfg_eval;
