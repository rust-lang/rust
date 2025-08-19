//@ aux-build: reexports-attrs.rs

#![crate_name = "foo"]

extern crate reexports_attrs;

//@ has 'foo/fn.f0.html' '//pre[@class="rust item-decl"]' '#[unsafe(no_mangle)]'
pub use reexports_attrs::f0;

//@ has 'foo/fn.f1.html' '//pre[@class="rust item-decl"]' '#[unsafe(link_section = ".here")]'
pub use reexports_attrs::f1;

//@ has 'foo/fn.f2.html' '//pre[@class="rust item-decl"]' '#[unsafe(export_name = "f2export")]'
pub use reexports_attrs::f2;

//@ has 'foo/enum.T0.html' '//pre[@class="rust item-decl"]' '#[repr(u8)]'
pub use reexports_attrs::T0;

//@ has 'foo/enum.T1.html' '//pre[@class="rust item-decl"]' '#[non_exhaustive]'
pub use reexports_attrs::T1;
