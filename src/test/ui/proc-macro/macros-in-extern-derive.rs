extern "C" {
    #[derive(Copy)] //~ ERROR `derive` may only be applied to structs, enums and unions
    fn f();
}

fn main() {}
