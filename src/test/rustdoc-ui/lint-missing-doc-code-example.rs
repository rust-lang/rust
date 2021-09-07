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
pub fn test() {
}

#[allow(missing_docs)]
pub mod module1 { //~ ERROR
}

#[allow(rustdoc::missing_doc_code_examples)]
/// doc
pub mod module2 {

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
pub union Union {
    /// Doc, but no code example and it's fine!
    a: i32,
    /// Doc, but no code example and it's fine!
    b: f32,
}

// no code example and it's fine!
impl Clone for Struct {
    fn clone(&self) -> Self {
        Self { field: self.field }
    }
}



/// doc
///
/// ```
/// println!("hello");
/// ```
#[derive(Clone)]
pub struct NiceStruct;

#[doc(hidden)]
pub mod foo {
    pub fn bar() {}
}

fn babar() {}


mod fofoo {
    pub fn tadam() {}
}
