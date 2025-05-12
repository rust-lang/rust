// rustfmt-normalize_comments: true
// rustfmt-format_code_in_doc_comments: true

// Do not unindent macro calls in comment with unformattable syntax.
//! ```rust
//! let x = 3  ;
//! some_macro!(pub fn fn foo() (
//!     println!("Don't unindent me!");
//! ));
//! ```

// Format items that appear as arguments of macro call.
//! ```rust
//! let x = 3;
//! some_macro!(
//!     pub fn foo() {
//!         println!("Don't unindent me!");
//!     }
//! );
//! ```
