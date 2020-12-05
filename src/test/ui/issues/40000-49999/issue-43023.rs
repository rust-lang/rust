struct S;

impl S {
    #[derive(Debug)] //~ ERROR `derive` may only be applied to structs, enums and unions
    fn f() {
        file!();
    }
}

trait Tr1 {
    #[derive(Debug)] //~ ERROR `derive` may only be applied to structs, enums and unions
    fn f();
}

trait Tr2 {
    #[derive(Debug)] //~ ERROR `derive` may only be applied to structs, enums and unions
    type F;
}

fn main() {}
