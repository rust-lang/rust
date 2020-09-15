// compile-flags: --document-private-items

#![deny(missing_docs)]
#![deny(missing_doc_code_examples)]

//! hello
//!
//! ```
//! let x = 0;
//! ```

fn private_fn() {}
//~^ ERROR
//~^^ ERROR
