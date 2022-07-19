// Test for the ICE in rust/83720
// A pub-in-private type re-exported under two different names shouldn't cause an error

#![no_core]
#![feature(no_core)]

// @is private_two_names.json "$.index[*][?(@.name=='style')].kind" \"module\"
// @is private_two_names.json "$.index[*][?(@.name=='style')].inner.is_stripped" "true"
mod style {
    // @has - "$.index[*](?(@.name=='Color'))"
    pub struct Color;
}

// @has - "$.index[*][?(@.kind=='import' && @.inner.name=='Color')]"
pub use style::Color;
// @has - "$.index[*][?(@.kind=='import' && @.inner.name=='Colour')]"
pub use style::Color as Colour;
