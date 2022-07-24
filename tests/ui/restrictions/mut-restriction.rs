#![feature(mut_restriction)]

pub mod foo {
    #[derive(Default)]
    pub struct Foo {
        pub mut(self) alpha: u8,
    }

    pub enum Bar {
        Beta(mut(self) u8),
    }

    impl Default for Bar {
        fn default() -> Self {
            Bar::Beta(0)
        }
    }
}

fn mut_direct(foo: &mut foo::Foo, bar: &mut foo::Bar) {
    foo.alpha = 1; //~ ERROR mutable use of restricted field
    match bar {
        foo::Bar::Beta(ref mut beta) => {} //~ ERROR mutable use of restricted field
    }
}

fn mut_ptr(foo: *mut foo::Foo) {
    // unsafe doesn't matter
    unsafe {
        (*foo).alpha = 1; //~ ERROR mutable use of restricted field
    }
}

fn main() {
    let mut foo = foo::Foo::default();
    let mut bar = foo::Bar::default();

    foo.alpha = 1; //~ ERROR mutable use of restricted field
    match bar {
        foo::Bar::Beta(ref mut beta) => {} //~ ERROR mutable use of restricted field
    }
    std::ptr::addr_of_mut!(foo.alpha); //~ ERROR mutable use of restricted field

    let _alpha = &mut foo.alpha; //~ ERROR mutable use of restricted field

    let mut closure = || {
        foo.alpha = 1; //~ ERROR mutable use of restricted field
    };

    // okay: the mutation occurs inside the function
    closure();
    mut_direct(&mut foo, &mut bar);
    mut_ptr(&mut foo as *mut _);

    // okay: this is the same as turning &T into &mut T, which is unsound
    unsafe { *(&foo.alpha as *const _ as *mut _) = 1; }
}
