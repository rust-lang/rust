// This test checks the "reexported from" link generation.

#![crate_name = "foo"]

mod a {
    pub fn foo() {}
}

pub mod b {
    pub fn blob() {}
}

// Reexport of a private item, no link generated.
//@ has 'foo/fn.bar.html'
//@ has - '//*[@class="reexport-origin"]' 'Reexported from crate::a::foo'
//@ count - '//*[@class="reexport-origin"]/a' 0
pub use crate::a::foo as bar;
#[doc(inline)]
// Reexport of a public item, link generated.
//@ has 'foo/fn.blob.html'
//@ has - '//*[@class="reexport-origin"]' 'Reexported from crate::b::blob'
//@ has - '//*[@class="reexport-origin"]/a[@href="b/fn.blob.html"]' 'crate::b::blob'
pub use crate::b::blob;
// Reexport of reexport (alloc -> std -> here).
//@ has 'foo/struct.Vec.html'
//@ has - '//*[@class="reexport-origin"]' 'Reexported from alloc::vec::Vec'
//@ has - '//*[@class="reexport-origin"]/a[@href="{{channel}}/alloc/vec/struct.Vec.html"]' \
//        'alloc::vec::Vec'
#[doc(inline)]
pub use std::vec::*;
// Reexport of a module, we check that its children also display the "reexported from".
//@ has 'foo/string/struct.String.html'
//@ has - '//*[@class="reexport-origin"]' 'Reexported from alloc::string::String'
//@ has - '//*[@class="reexport-origin"]/a[@href="{{channel}}/alloc/string/struct.String.html"]' \
//        'alloc::string::String'
// While we're at it, we also check that the `string` module has the "reexported from".
//@ has 'foo/string/index.html'
//@ has - '//*[@class="reexport-origin"]' 'Reexported from alloc::string'
//@ has - '//*[@class="reexport-origin"]/a[@href="{{channel}}/alloc/string/index.html"]' \
//        'alloc::string'
#[doc(inline)]
pub use std::string;

//@ has 'foo/macro.my_println.html'
//@ has - '//*[@class="reexport-origin"]' 'Reexported from std::println'
//@ has - '//*[@class="reexport-origin"]/a[@href="{{channel}}/std/macro.println.html"]' \
//        'std::println'
pub use std::println as my_println;
