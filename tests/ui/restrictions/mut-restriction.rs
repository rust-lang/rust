// aux-build: external-mut-restriction.rs

#![feature(mut_restriction)]

extern crate external_mut_restriction;

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
    foo.alpha = 1; //~ ERROR field cannot be mutated outside `foo`
    match bar {
        foo::Bar::Beta(ref mut beta) => {} //~ ERROR field cannot be mutated outside `foo`
    }
}

fn mut_ptr(foo: *mut foo::Foo) {
    // unsafe doesn't matter
    unsafe {
        (*foo).alpha = 1; //~ ERROR field cannot be mutated outside `foo`
    }
}

fn main() {
    let mut foo = foo::Foo::default();
    let mut bar = foo::Bar::default();

    foo.alpha = 1; //~ ERROR field cannot be mutated outside `foo`
    match bar {
        foo::Bar::Beta(ref mut beta) => {} //~ ERROR field cannot be mutated outside `foo`
    }
    std::ptr::addr_of_mut!(foo.alpha); //~ ERROR field cannot be mutated outside `foo`

    let _alpha = &mut foo.alpha; //~ ERROR field cannot be mutated outside `foo`

    let mut closure = || {
        foo.alpha = 1; //~ ERROR field cannot be mutated outside `foo`
    };

    // okay: the mutation occurs inside the function
    closure();
    mut_direct(&mut foo, &mut bar);
    mut_ptr(&mut foo as *mut _);

    // undefined behavior, but not a compile error (it is the same as turning &T into &mut T)
    unsafe { *(&foo.alpha as *const _ as *mut _) = 1; }

    let mut external_top_level = external_mut_restriction::TopLevel::new();
    external_top_level.alpha = 1; //~ ERROR field cannot be mutated outside
    //FIXME~^ ERROR field cannot be mutated outside `external_mut_restriction`

    let mut external_inner = external_mut_restriction::inner::Inner::new();
    external_inner.beta = 1; //~ ERROR field cannot be mutated outside
    //FIXME~^ ERROR field cannot be mutated outside `external_mut_restriction`
}
