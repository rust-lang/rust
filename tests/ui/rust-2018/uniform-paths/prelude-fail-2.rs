//@ edition:2018

// Built-in attribute
use inline as imported_inline;
mod builtin {
    pub use inline as imported_inline;
}

#[imported_inline] //~ ERROR cannot use a built-in attribute through an import
#[builtin::imported_inline] //~ ERROR cannot use a built-in attribute through an import
fn main() {}
