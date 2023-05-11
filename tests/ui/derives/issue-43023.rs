struct S;

impl S {
    #[derive(Debug)] //~ ERROR `derive` may only be applied to `struct`s, `enum`s and `union`s
    fn f() {
        file!();
    }
}

trait Tr1 {
    #[derive(Debug)] //~ ERROR `derive` may only be applied to `struct`s, `enum`s and `union`s
    fn f();
}

trait Tr2 {
    #[derive(Debug)] //~ ERROR `derive` may only be applied to `struct`s, `enum`s and `union`s
    type F;
}

fn main() {}
