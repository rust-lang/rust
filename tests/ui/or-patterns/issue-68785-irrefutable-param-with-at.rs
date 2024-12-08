//@ check-pass

enum MyEnum {
    FirstCase(u8),
    OtherCase(u16),
}

fn my_fn(x @ (MyEnum::FirstCase(_) | MyEnum::OtherCase(_)): MyEnum) {}

fn main() {
    my_fn(MyEnum::FirstCase(0));
}
