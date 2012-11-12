mod a {
	pub trait Foo {
		static pub fn foo() -> self;
	}

	impl int : Foo {
		static pub fn foo() -> int {
			3
		}
	}
	
	impl uint : Foo {
		static pub fn foo() -> uint {
			5u
		}
	}
}

fn main() {
	let x: int = a::Foo::foo();
	let y: uint = a::Foo::foo();
	assert x == 3;
	assert y == 5;
}

