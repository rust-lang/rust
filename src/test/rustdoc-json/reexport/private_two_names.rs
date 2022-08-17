// ignore-tidy-linelength

// Test for the ICE in https://github.com/rust-lang/rust/issues/83720
// A pub-in-private type re-exported under two different names shouldn't cause an error

#![no_core]
#![feature(no_core)]

// @is "$.index[*][?(@.name=='style')].kind" \"module\"
// @is "$.index[*][?(@.name=='style')].inner.is_stripped" "true"
mod style {
    // @set color_struct_id = "$.index[*][?(@.kind=='struct' && @.name=='Color')].id"
    pub struct Color;
}

// @is "$.index[*][?(@.kind=='import' && @.inner.name=='Color')].inner.id" $color_struct_id
// @set color_export_id = "$.index[*][?(@.kind=='import' && @.inner.name=='Color')].id"
pub use style::Color;
// @is "$.index[*][?(@.kind=='import' && @.inner.name=='Colour')].inner.id" $color_struct_id
// @set colour_export_id = "$.index[*][?(@.kind=='import' && @.inner.name=='Colour')].id"
pub use style::Color as Colour;

// @ismany "$.index[*][?(@.name=='private_two_names')].inner.items[*]" $color_export_id $colour_export_id
