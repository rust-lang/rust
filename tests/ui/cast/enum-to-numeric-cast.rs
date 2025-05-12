// Tests that `as` casts from enums to numeric types succeed
// only if the enum type is "unit-only" or "fieldless" as
// described here: https://doc.rust-lang.org/reference/items/enumerations.html#casting

pub enum UnitOnly {
    Foo,
    Bar,
    Baz,
}

pub enum Fieldless {
    Tuple(),
    Struct{},
    Unit,
}

pub enum NotUnitOnlyOrFieldless {
    Foo,
    Bar(u8),
    Baz
}

fn main() {
    let unit_only = UnitOnly::Foo;

    let _ = unit_only as isize;
    let _ = unit_only as i32;
    let _ = unit_only as usize;
    let _ = unit_only as u32;


    let fieldless = Fieldless::Struct{};

    let _ = fieldless as isize;
    let _ = fieldless as i32;
    let _ = fieldless as usize;
    let _ = fieldless as u32;


    let not_unit_only_or_fieldless = NotUnitOnlyOrFieldless::Foo;

    let _ = not_unit_only_or_fieldless as isize; //~ ERROR non-primitive cast: `NotUnitOnlyOrFieldless` as `isize`
    let _ = not_unit_only_or_fieldless as i32; //~ ERROR non-primitive cast: `NotUnitOnlyOrFieldless` as `i32`
    let _ = not_unit_only_or_fieldless as usize; //~ ERROR non-primitive cast: `NotUnitOnlyOrFieldless` as `usize`
    let _ = not_unit_only_or_fieldless as u32; //~ ERROR non-primitive cast: `NotUnitOnlyOrFieldless` as `u32`
}
