// Regression test for <https://github.com/rust-lang/rust/issues/158174>.
#![crate_name = "foo"]

pub struct S0;
pub struct S1;
pub struct S2;
pub struct S3;

impl S0 { pub fn foo(&self) {} }
impl S1 { pub fn foo(&self) {} }
impl S2 { pub fn foo(&self) {} }
impl S3 { pub fn foo(&self) {} }

impl std::ops::Deref for S0 { type Target = S1; fn deref(&self) -> &S1 { &S1 } }
impl std::ops::Deref for S1 { type Target = S2; fn deref(&self) -> &S2 { &S2 } }
impl std::ops::Deref for S2 { type Target = S3; fn deref(&self) -> &S3 { &S3 } }

//@ has foo/struct.S0.html '//*[@class="sidebar-elems"]//section//li/a[@href="#method.foo"]' 'foo'
//@ has foo/struct.S0.html '//*[@class="sidebar-elems"]//section//li/a[@href="#method.foo-1"]' 'foo'
//@ has foo/struct.S0.html '//*[@class="sidebar-elems"]//section//li/a[@href="#method.foo-2"]' 'foo'
//@ has foo/struct.S0.html '//*[@class="sidebar-elems"]//section//li/a[@href="#method.foo-3"]' 'foo'
