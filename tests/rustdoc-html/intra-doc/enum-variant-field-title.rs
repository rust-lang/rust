// This test ensures that the generated "title" attribute for field of enum variant
// is correctly generated and includes the variant's name.
// Regression test for <https://github.com/rust-lang/rust/issues/161028>.

#![crate_name = "foo"]

//@ has 'foo/enum.Enum.html'
//@ has - '//*[@class="docblock"]//a[@title="field foo::Enum::Variant::field"]' 'Enum::Variant::field'

/// [Enum::Variant::field]
pub enum Enum {
    Variant { field: i32 }
}
