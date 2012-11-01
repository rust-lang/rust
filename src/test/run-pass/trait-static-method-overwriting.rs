mod base {
    pub trait HasNew<T> {
        static pure fn new() -> T;
    }

    pub struct Foo {
        dummy: (),
    }

    pub impl Foo : base::HasNew<Foo> {
        static pure fn new() -> Foo {
			unsafe { io::println("Foo"); }
            Foo { dummy: () }
        }
    }

    pub struct Bar {
        dummy: (),
    }

    pub impl Bar : base::HasNew<Bar> {
        static pure fn new() -> Bar {
			unsafe { io::println("Bar"); }
            Bar { dummy: () }
        }
    }
}

fn main() {
    let f: base::Foo = base::new::<base::Foo, base::Foo>();
	let b: base::Bar = base::new::<base::Bar, base::Bar>();
}
