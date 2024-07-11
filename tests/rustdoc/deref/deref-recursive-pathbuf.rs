// #26207: Show all methods reachable via Deref impls, recursing through multiple dereferencing
// levels and across multiple crates.
// For `Deref` on non-foreign types, look at `deref-recursive.rs`.

//@ has 'foo/struct.Foo.html'
//@ has '-' '//*[@id="deref-methods-PathBuf"]' 'Methods from Deref<Target = PathBuf>'
//@ has '-' '//*[@class="impl-items"]//*[@id="method.as_path"]' 'pub fn as_path(&self)'
//@ has '-' '//*[@id="deref-methods-Path"]' 'Methods from Deref<Target = Path>'
//@ has '-' '//*[@class="impl-items"]//*[@id="method.exists"]' 'pub fn exists(&self)'
//@ has '-' '//div[@class="sidebar-elems"]//h3/a[@href="#deref-methods-PathBuf"]' 'Methods from Deref<Target=PathBuf>'
//@ has '-' '//*[@class="sidebar-elems"]//*[@class="block deref-methods"]//a[@href="#method.as_path"]' 'as_path'
//@ has '-' '//div[@class="sidebar-elems"]//h3/a[@href="#deref-methods-Path"]' 'Methods from Deref<Target=Path>'
//@ has '-' '//*[@class="sidebar-elems"]//*[@class="block deref-methods"]//a[@href="#method.exists"]' 'exists'

#![crate_name = "foo"]

use std::ops::Deref;
use std::path::PathBuf;

pub struct Foo(PathBuf);

impl Deref for Foo {
    type Target = PathBuf;
    fn deref(&self) -> &PathBuf { &self.0 }
}
