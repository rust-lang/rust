// TODO: add header.
// do I need to add this to some file?

use std::ops::Deref;

struct Foo {
	inner:		Bar,
}

struct Bar;

impl Foo {
	pub fn foo_method(&self) {
		/* .. */
	}
}

impl Bar {
	pub fn bar_method(&self) {
		/* .. */
	}
}

impl Deref for Foo {
	type Target = Bar;

	fn deref<'a>(&'a self) -> &'a Self::Target {
		&self.inner
	}
}

impl Deref for Bar {
	type Target = Foo;

	fn deref<'a>(&'a self) -> &'a Self::Target {
		panic!()
	}
}

fn main() {
	let foo = Foo { inner: Bar, };
	let bar = Bar;
	
	foo.bar_method();	// should compile and execute
}
