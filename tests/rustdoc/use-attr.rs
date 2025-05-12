//@ edition:2018

// ICE when rustdoc encountered a use statement of a non-macro attribute (see #58054)

//@ has use_attr/index.html
//@ has - '//code' 'pub use proc_macro_attribute'
pub use proc_macro_attribute;
use proc_macro_derive;
