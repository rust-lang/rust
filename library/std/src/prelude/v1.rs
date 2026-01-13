//! The first version of the prelude of The Rust Standard Library.
//!
//! See the [module-level documentation](super) for more.

#![stable(feature = "rust1", since = "1.0.0")]

// No formatting: this file is nothing but re-exports, and their order is worth preserving.
#![cfg_attr(rustfmt, rustfmt::skip)]

// Re-exported core operators
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::marker::{Send, Sized, Sync, Unpin};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::ops::{Drop, Fn, FnMut, FnOnce};
#[stable(feature = "async_closure", since = "1.85.0")]
#[doc(no_inline)]
pub use crate::ops::{AsyncFn, AsyncFnMut, AsyncFnOnce};

// Re-exported functions
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::mem::drop;
#[stable(feature = "size_of_prelude", since = "1.80.0")]
#[doc(no_inline)]
pub use crate::mem::{align_of, align_of_val, size_of, size_of_val};

// Re-exported types and traits
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::convert::{AsMut, AsRef, From, Into};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::iter::{DoubleEndedIterator, ExactSizeIterator};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::iter::{Extend, IntoIterator, Iterator};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::option::Option::{self, None, Some};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::result::Result::{self, Err, Ok};

// Re-exported built-in macros and traits
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[doc(no_inline)]
#[expect(deprecated)]
pub use core::prelude::v1::{
    assert, assert_eq, assert_ne, cfg, column, compile_error, concat, debug_assert, debug_assert_eq,
    debug_assert_ne, env, file, format_args, include, include_bytes, include_str, line, matches,
    module_path, option_env, stringify, todo, r#try, unimplemented, unreachable, write,
    writeln, Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd,
};

#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[doc(no_inline)]
pub use crate::{
    dbg, eprint, eprintln, format, is_x86_feature_detected, print, println, thread_local
};

// These macros need special handling, so that we don't export them *and* the modules of the same
// name. We only want the macros in the prelude so we shadow the original modules with private
// modules with the same names.
mod ambiguous_macros_only {
    #[expect(hidden_glob_reexports)]
    mod vec {}
    #[expect(hidden_glob_reexports)]
    mod panic {}
    // Building std without the expect exported_private_dependencies will create warnings, but then
    // clippy claims its a useless_attribute. So silence both.
    #[expect(clippy::useless_attribute)]
    #[expect(exported_private_dependencies)]
    #[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
    pub use crate::*;
}
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[doc(no_inline)]
pub use self::ambiguous_macros_only::{vec, panic};

#[unstable(feature = "cfg_select", issue = "115585")]
#[doc(no_inline)]
pub use core::prelude::v1::cfg_select;

#[unstable(
    feature = "concat_bytes",
    issue = "87555",
    reason = "`concat_bytes` is not stable enough for use and is subject to change"
)]
#[doc(no_inline)]
pub use core::prelude::v1::concat_bytes;

#[unstable(feature = "const_format_args", issue = "none")]
#[doc(no_inline)]
pub use core::prelude::v1::const_format_args;

#[unstable(
    feature = "log_syntax",
    issue = "29598",
    reason = "`log_syntax!` is not stable enough for use and is subject to change"
)]
#[doc(no_inline)]
pub use core::prelude::v1::log_syntax;

#[unstable(
    feature = "trace_macros",
    issue = "29598",
    reason = "`trace_macros` is not stable enough for use and is subject to change"
)]
#[doc(no_inline)]
pub use core::prelude::v1::trace_macros;

// Do not `doc(no_inline)` so that they become doc items on their own
// (no public module for them to be re-exported from).
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
pub use core::prelude::v1::{
    alloc_error_handler, bench, derive, global_allocator, test, test_case,
};

#[unstable(feature = "derive_const", issue = "118304")]
pub use core::prelude::v1::derive_const;

// Do not `doc(no_inline)` either.
#[unstable(
    feature = "cfg_accessible",
    issue = "64797",
    reason = "`cfg_accessible` is not fully implemented"
)]
pub use core::prelude::v1::cfg_accessible;

// Do not `doc(no_inline)` either.
#[unstable(
    feature = "cfg_eval",
    issue = "82679",
    reason = "`cfg_eval` is a recently implemented feature"
)]
pub use core::prelude::v1::cfg_eval;

// Do not `doc(no_inline)` either.
#[unstable(
    feature = "type_ascription",
    issue = "23416",
    reason = "placeholder syntax for type ascription"
)]
pub use core::prelude::v1::type_ascribe;

// Do not `doc(no_inline)` either.
#[unstable(
    feature = "deref_patterns",
    issue = "87121",
    reason = "placeholder syntax for deref patterns"
)]
pub use core::prelude::v1::deref;

// Do not `doc(no_inline)` either.
#[unstable(
    feature = "type_alias_impl_trait",
    issue = "63063",
    reason = "`type_alias_impl_trait` has open design concerns"
)]
pub use core::prelude::v1::define_opaque;

#[unstable(feature = "extern_item_impls", issue = "125418")]
pub use core::prelude::v1::{eii, unsafe_eii};

#[unstable(feature = "eii_internals", issue = "none")]
pub use core::prelude::v1::eii_declaration;

// The file so far is equivalent to core/src/prelude/v1.rs. It is duplicated
// rather than glob imported because we want docs to show these re-exports as
// pointing to within `std`.
// Below are the items from the alloc crate.

#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::borrow::ToOwned;
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::boxed::Box;
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::string::{String, ToString};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use crate::vec::Vec;
