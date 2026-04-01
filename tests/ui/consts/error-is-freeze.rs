// Make sure we treat the error type as freeze to suppress useless errors.

struct MyStruct {
    foo: Option<UndefinedType>,
    //~^ ERROR cannot find type `UndefinedType` in this scope
}
impl MyStruct {
    pub const EMPTY_REF: &'static Self = &Self::EMPTY;
    pub const EMPTY: Self = Self {
        foo: None,
    };
}

fn main() {}
