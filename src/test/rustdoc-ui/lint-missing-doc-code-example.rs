#![deny(missing_docs)]
#![deny(missing_doc_code_examples)]

//! crate level doc
//! ```
//! println!("hello"):
//! ```


/// doc
///
/// ```
/// println!("hello");
/// ```
fn test() {
}

#[allow(missing_docs)]
mod module1 { //~ ERROR
}

#[allow(missing_doc_code_examples)]
/// doc
mod module2 {

  /// doc
  pub fn test() {}
}

/// doc
///
/// ```
/// println!("hello");
/// ```
pub mod module3 {

  /// doc
  //~^ ERROR
  pub fn test() {}
}
