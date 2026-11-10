#![feature(default_field_values)]

#[derive(Default)]
#[non_exhaustive]
//~^ ERROR `#[non_exhaustive]` can't be used to annotate structs with default field values
struct Struct {
    field: i32 = 42 + 3,
}

enum Enum {
    #[non_exhaustive]
    //~^ ERROR `#[non_exhaustive]` can't be used to annotate structs with default field values
    Variant {
        field: i32 = 24,
    }
}

#[derive(Default)]
enum EnumWithDefault {
    #[non_exhaustive]
    //~^ ERROR `#[non_exhaustive]` can't be used to annotate structs with default field values
    #[default] //~ ERROR default variant must be exhaustive
    Variant {
        field: i32 = 42 + 3,
    }
}


fn main () {}
