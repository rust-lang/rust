#![deny(missing_docs)]
#![deny(rustdoc::missing_doc_code_examples)]

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

#[allow(rustdoc::missing_doc_code_examples)]
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

/// Doc, but no code example and it's fine!
pub const Const: u32 = 0;
/// Doc, but no code example and it's fine!
pub static Static: u32 = 0;
/// Doc, but no code example and it's fine!
pub type Type = u32;

/// Doc
//~^ ERROR
pub struct Struct {
    /// Doc, but no code example and it's fine!
    pub field: u32,
}

/// Doc
//~^ ERROR
pub enum Enum {
    /// Doc, but no code example and it's fine!
    X,
}

/// Doc
//~^ ERROR
#[repr(C)]
union Union {
    /// Doc, but no code example and it's fine!
    a: i32,
    /// Doc, but no code example and it's fine!
    b: f32,
}
