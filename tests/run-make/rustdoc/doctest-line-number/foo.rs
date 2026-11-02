//! This library exposes foo for fooing:
#![cfg_attr(not(feature = "foo"), doc = "```ignore")]
#![cfg_attr(feature = "foo", doc = "```rust")]
//! let x = 12;
//! ```
//!
// But baring is also possible:
#![cfg_attr(not(feature = "bar"), doc = "```ignore")]
#![cfg_attr(feature = "bar", doc = "```rust")]
//! let x = 12;
//! ```
//!
// A code comment!
//! ```rust
//! let x = 12;
//! ```
//!
