// `#![derive]` raises errors when it occurs at contexts other than ADT
// definitions.

#[derive(Debug)]
//~^ ERROR `derive` may only be applied to structs, enums and unions
mod derive {
    mod inner { #![derive(Debug)] }
    //~^ ERROR `derive` may only be applied to structs, enums and unions

    #[derive(Debug)]
    //~^ ERROR `derive` may only be applied to structs, enums and unions
    fn derive() { }

    #[derive(Copy, Clone)] // (can't derive Debug for unions)
    union U { f: i32 }

    #[derive(Debug)]
    struct S;

    #[derive(Debug)]
    enum E { }

    #[derive(Debug)]
    //~^ ERROR `derive` may only be applied to structs, enums and unions
    type T = S;

    #[derive(Debug)]
    //~^ ERROR `derive` may only be applied to structs, enums and unions
    impl S { }
}

fn main() {}
