// This test ensures a few things around the `rustc_must_implement_one_of` attribute:
// 1. The listed methods are links
// 2. It gets a heading in the sidebar even if there is no "required methods"
// Regression test for <https://github.com/rust-lang/rust/issues/163675>.

#![allow(internal_features)]
#![feature(rustc_attrs)]

#![crate_name = "foo"]

//@ has 'foo/trait.Read.html'
//@ has - '//*[@class="stab must_implement"]/a[@href="#method.read"]' 'read'
//@ has - '//*[@class="stab must_implement"]/a[@href="#method.read_buf"]' 'read_buf'

// There should be 3 headings in the sidebar: Required/Provided methods and "Implementors".
//@ count - '//*[@id="rustdoc-toc"]/h3' 3
//@ has - '//*[@id="rustdoc-toc"]/h3/a[@href="#required-methods"]' 'Required Methods'
//@ has - '//*[@id="rustdoc-toc"]/h3/a[@href="#provided-methods"]' 'Provided Methods'
//@ has - '//*[@id="rustdoc-toc"]/h3/a[@href="#implementors"]' 'Implementors'

#[rustc_must_implement_one_of(read_buf, read)]
pub trait Read {
    fn read_buf(&self) {}
    fn read(&self) {}
}

//@ has 'foo/trait.Tadam.html'
// There should still be 3 headings in the sidebar: Required/Provided methods and "Implementors".
//@ count - '//*[@id="rustdoc-toc"]/h3' 3
//@ has - '//*[@id="rustdoc-toc"]/h3/a[@href="#required-methods"]' 'Required Methods'
//@ has - '//*[@id="rustdoc-toc"]/h3/a[@href="#provided-methods"]' 'Provided Methods'
//@ has - '//*[@id="rustdoc-toc"]/h3/a[@href="#implementors"]' 'Implementors'

#[rustc_must_implement_one_of(read_buf, read)]
pub trait Tadam {
    fn read_buf(&self) {}
    fn read(&self) {}
    fn foo(&self) {}
}

//@ has 'foo/trait.Other.html'
// There should be 2 headings in the sidebar: Provided methods and "Implementors".
//@ count - '//*[@id="rustdoc-toc"]/h3' 2
//@ has - '//*[@id="rustdoc-toc"]/h3/a[@href="#provided-methods"]' 'Provided Methods'
//@ has - '//*[@id="rustdoc-toc"]/h3/a[@href="#implementors"]' 'Implementors'

pub trait Other {
    fn read_buf(&self) {}
    fn read(&self) {}
}
