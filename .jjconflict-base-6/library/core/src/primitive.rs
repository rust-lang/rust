//! This module reexports the primitive types to allow usage that is not
//! possibly shadowed by other declared types.
//!
//! This is normally only useful in macro generated code.
//!
//! An example of this is when generating a new struct and an impl for it:
//!
//! ```rust,compile_fail
//! pub struct bool;
//!
//! impl QueryId for bool {
//!     const SOME_PROPERTY: bool = true;
//! }
//!
//! # trait QueryId { const SOME_PROPERTY: ::core::primitive::bool; }
//! ```
//!
//! Note that the `SOME_PROPERTY` associated constant would not compile, as its
//! type `bool` refers to the struct, rather than to the primitive bool type.
//!
//! A correct implementation could look like:
//!
//! ```rust
//! # #[allow(non_camel_case_types)]
//! pub struct bool;
//!
//! impl QueryId for bool {
//!     const SOME_PROPERTY: ::core::primitive::bool = true;
//! }
//!
//! # trait QueryId { const SOME_PROPERTY: ::core::primitive::bool; }
//! ```
//!
//! We also used `::core` instead of `core`, because `core` can be
//! shadowed, too. Paths, starting with `::`, are searched in
//! the [extern prelude] since Edition 2018.
//!
//! [extern prelude]: https://doc.rust-lang.org/nightly/reference/names/preludes.html#extern-prelude

#[stable(feature = "core_primitive", since = "1.43.0")]
pub use bool;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use char;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use f32;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use f64;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use i8;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use i16;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use i32;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use i64;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use i128;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use isize;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use str;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use u8;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use u16;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use u32;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use u64;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use u128;
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use usize;
