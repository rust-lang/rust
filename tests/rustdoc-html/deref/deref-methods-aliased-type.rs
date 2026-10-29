// Ensures that aliased types get the "Methods from Deref" but only when it matches
// the aliased type.
// Regression test for <https://github.com/rust-lang/rust/issues/134868>.

// ignore-tidy-file-linelength

#![crate_name = "foo"]

use std::ops::{Deref, DerefMut};

pub struct Target;

impl Target {
    pub fn target_ref(&self) {}
    pub fn target_mut(&mut self) {}
}

pub struct Other;

impl Other {
    pub fn other_ref(&self) {}
    pub fn other_mut(&mut self) {}
}

pub struct Wrapper<T>(T);

impl Deref for Wrapper<u8> {
    type Target = Target;
    fn deref(&self) -> &Target {
        &Target
    }
}

impl Deref for Wrapper<u16> {
    type Target = Other;
    fn deref(&self) -> &Other {
        &Other
    }
}

impl DerefMut for Wrapper<u16> {
    fn deref_mut(&mut self) -> &mut Other {
        &mut Other
    }
}

//@ has 'foo/type.Byte.html'
// Only one deref.
//@ count - '//h2[@id="deref-methods-Target"]' 1
// Which makes two section headers. One for the "Aliased Type" and one for "Methods from Deref<>".
//@ count - '//*[@class="section-header"]' 2
//@ has - '//*[@class="section-header"]' 'Aliased Type'
//@ has - '//*[@class="section-header"]' 'Methods from Deref<Target = Target>'
// Only one method from the deref (not the one taking `&mut` since `DerefMut` is not
// implemented for `Wrapper<u8>`.
//@ count - '//*[@id="deref-methods-Target-1"]/*[@class="method"]' 1
//@ has - '//*[@id="method.target_ref"]//h4' 'pub fn target_ref(&self)'

// Same checks but in the sidebar.
//@ count - '//*[@class="sidebar-elems"]//*[@class="block deref-methods"]//a' 1
//@ has - '//*[@class="sidebar-elems"]//*[@class="block deref-methods"]//a[@href="#method.target_ref"]' 'target_ref'
pub type Byte = Wrapper<u8>;

//@ has 'foo/type.Word.html'
// There should be only one `Deref` block (for both `Deref` and `DerefMut`).
//@ count - '//h2[@id="deref-methods-Other"]' 1
// Which makes two section headers. One for the "Aliased Type" and one for "Methods from Deref<>".
//@ count - '//*[@class="section-header"]' 2
//@ has - '//*[@class="section-header"]' 'Aliased Type'
//@ has - '//*[@class="section-header"]' 'Methods from Deref<Target = Other>'
// Since it implements both `Deref` and `DerefMut`, we should see both methods.
//@ count - '//*[@id="deref-methods-Other-1"]/*[@class="method"]' 2
//@ has - '//*[@id="method.other_ref"]//h4' 'pub fn other_ref(&self)'
//@ has - '//*[@id="method.other_mut"]//h4' 'pub fn other_mut(&mut self)'

// Same checks but in the sidebar.
//@ count - '//*[@class="sidebar-elems"]//*[@class="block deref-methods"]//a' 2
//@ has - '//*[@class="sidebar-elems"]//*[@class="block deref-methods"]//a[@href="#method.other_ref"]' 'other_ref'
//@ has - '//*[@class="sidebar-elems"]//*[@class="block deref-methods"]//a[@href="#method.other_mut"]' 'other_mut'
pub type Word = Wrapper<u16>;

//@ has 'foo/type.Dword.html'
// There is no `Deref` implementation for `Wrapper<u32>` so we confirm that.
// Which makes two section headers. One for the "Aliased Type" and one for "Methods from Deref<>".
//@ count - '//*[@class="section-header"]' 1
//@ has - '//*[@class="section-header"]' 'Aliased Type'
pub type Dword = Wrapper<u32>;
