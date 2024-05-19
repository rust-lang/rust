extern "C" {
    #[derive(Copy)] //~ ERROR `derive` may only be applied to `struct`s, `enum`s and `union`s
    fn f();
}

fn main() {}
