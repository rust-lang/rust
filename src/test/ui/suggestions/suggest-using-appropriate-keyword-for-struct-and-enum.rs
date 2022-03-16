pub struct NotStruct {
    Variant
} //~ ERROR expected `:`, found `}`

pub struct NotStruct {
    field String //~ ERROR expected `:`, found `String`
}

pub enum NotEnum {
    field: u8 //~ ERROR the enum cannot have a struct field declaration
}

fn main() {}
