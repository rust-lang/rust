pub trait Whatever {
    type Foo;

    fn method() {}
}

pub struct Struct;

impl Whatever for Struct {
    type Foo = u8;
}
