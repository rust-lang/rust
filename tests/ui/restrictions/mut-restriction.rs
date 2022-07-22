#![feature(restrictions)]

pub mod foo {
    #[derive(Default)]
    pub struct Foo {
        pub mut(self) alpha: u8,
    }
}

fn change_alpha(foo: &mut foo::Foo) {
    foo.alpha = 1; //~ ERROR mutable use of restricted field
}

fn change_alpha_ptr(foo: *mut foo::Foo) {
    // unsafe doesn't matter
    unsafe {
        (*foo).alpha = 1; //~ ERROR mutable use of restricted field
    }
}

fn main() {
    let mut foo = foo::Foo::default();
    foo.alpha = 1; //~ ERROR mutable use of restricted field
    std::ptr::addr_of_mut!(foo.alpha); //~ ERROR mutable use of restricted field

    let _alpha = &mut foo.alpha; //~ ERROR mutable use of restricted field

    let mut closure = || {
        foo.alpha = 1; //~ ERROR mutable use of restricted field
    };

    // okay: the mutation occurs inside the function
    closure();
    change_alpha(&mut foo);
    change_alpha_ptr(&mut foo as *mut _);

    // okay: this is the same as turning &T into &mut T, which is unsound
    unsafe { *(&foo.alpha as *const _ as *mut _) = 1; }
}
