// Test for the ICE in https://github.com/rust-lang/rust/issues/83720
// A pub-in-private type re-exported under two different names shouldn't cause an error

//@ !has "$.index[?(@.name=='style')]"
mod style {
    //@ set color_struct_id = "$.index[?(@.inner.struct && @.name=='Color')].id"
    pub struct Color;
}

//@ is "$.index[?(@.docs=='First re-export')].inner.use.id" $color_struct_id
//@ is "$.index[?(@.docs=='First re-export')].inner.use.name" \"Color\"
//@ set color_export_id = "$.index[?(@.docs=='First re-export')].id"
/// First re-export
pub use style::Color;
//@ is "$.index[?(@.docs=='Second re-export')].inner.use.id" $color_struct_id
//@ is "$.index[?(@.docs=='Second re-export')].inner.use.name" \"Colour\"
//@ set colour_export_id = "$.index[?(@.docs=='Second re-export')].id"
/// Second re-export
pub use style::Color as Colour;

//@ ismany "$.index[?(@.name=='private_two_names')].inner.module.items[*]" $color_export_id $colour_export_id
