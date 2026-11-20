//@ aux-build: bar.rs
//@ aux-build: foo.rs
//@ build-pass

#![deny(exported_private_dependencies)]

// Ensure the libbar.rlib is loaded first. If the command line parameter `--extern foo` does not
// exist, previous version would fail to compile
#![crate_type = "rlib"]
extern crate bar;
extern crate foo;
pub fn baz() -> (Option<foo::Foo>, Option<bar::Bar>) { (None, None) }
