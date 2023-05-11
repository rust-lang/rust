// Issue #54478: regression test showing that we can demonstrate
// `#[global_allocator]` in code blocks built by `rustdoc`.
//
// ## Background
//
// Changes in lang-item visibility injected failures that were only
// exposed when compiling with `-C prefer-dynamic`. But `rustdoc` used
// `-C prefer-dynamic` (and had done so for years, for reasons we did
// not document at that time).
//
// Rather than try to revise the visbility semanics, we instead
// decided to change `rustdoc` to behave more like the compiler's
// default setting, by leaving off `-C prefer-dynamic`.

// compile-flags:--test

//! This is a doc comment
//!
//! ```rust
//! use std::alloc::*;
//!
//! #[global_allocator]
//! static ALLOC: A = A;
//!
//! static mut HIT: bool = false;
//!
//! struct A;
//!
//! unsafe impl GlobalAlloc for A {
//!     unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
//!         HIT = true;
//!         System.alloc(layout)
//!     }
//!     unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
//!         System.dealloc(ptr, layout);
//!     }
//! }
//!
//! fn main() {
//!     assert!(unsafe { HIT });
//! }
//! ```
