//@ run-pass

#[non_exhaustive]
pub enum NonExhaustiveEnum {
    Unit,
    Tuple(u32),
    Struct { field: u32 }
}

fn main() {
    let enum_unit = NonExhaustiveEnum::Unit;

    match enum_unit {
        NonExhaustiveEnum::Unit => "first",
        NonExhaustiveEnum::Tuple(_) => "second",
        NonExhaustiveEnum::Struct { .. } => "third",
    };
}
