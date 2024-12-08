// `#![derive]` raises errors when it occurs at contexts other than ADT
// definitions.

#[derive(Debug)]
//~^ ERROR `derive` may only be applied to `struct`s, `enum`s and `union`s
mod derive {
    mod inner { #![derive(Debug)] }
    //~^ ERROR `derive` may only be applied to `struct`s, `enum`s and `union`s
    //~| ERROR inner macro attributes are unstable

    #[derive(Debug)]
    //~^ ERROR `derive` may only be applied to `struct`s, `enum`s and `union`s
    fn derive() { }

    #[derive(Copy, Clone)] // (can't derive Debug for unions)
    union U { f: i32 }

    #[derive(Debug)]
    struct S;

    #[derive(Debug)]
    enum E { }

    #[derive(Debug)]
    //~^ ERROR `derive` may only be applied to `struct`s, `enum`s and `union`s
    type T = S;

    #[derive(Debug)]
    //~^ ERROR `derive` may only be applied to `struct`s, `enum`s and `union`s
    impl S { }
}

fn main() {}
