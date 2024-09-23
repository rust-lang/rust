#![allow(dangling_pointers_from_temporaries)]

fn main() {
    dbg!(String::new().as_ptr());
    // ^ no error

    #[deny(dangling_pointers_from_temporaries)]
    {
        dbg!(String::new().as_ptr());
        //~^ ERROR getting a pointer from a temporary `String` will result in a dangling pointer
    }
    S.foo()
}

struct S;

impl S {
    #[warn(dangling_pointers_from_temporaries)]
    fn foo(self) {
        dbg!(String::new().as_ptr());
        //~^ WARNING getting a pointer from a temporary `String` will result in a dangling pointer
    }
}
