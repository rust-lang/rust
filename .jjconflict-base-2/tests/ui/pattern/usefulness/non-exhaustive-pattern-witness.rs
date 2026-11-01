struct Foo {
    first: bool,
    second: Option<[usize; 4]>,
}

fn struct_with_a_nested_enum_and_vector() {
    match (Foo { first: true, second: None }) {
        //~^ ERROR non-exhaustive patterns: `Foo { first: false, second: Some([0_usize, _, _, _]) }` and `Foo { first: false, second: Some([2_usize.., _, _, _]) }` not covered
        Foo { first: true, second: None } => (),
        Foo { first: true, second: Some(_) } => (),
        Foo { first: false, second: None } => (),
        Foo { first: false, second: Some([1, 2, 3, 4]) } => (),
    }
}

enum Color {
    Red,
    Green,
    CustomRGBA { a: bool, r: u8, g: u8, b: u8 },
}

fn enum_with_single_missing_variant() {
    match Color::Red {
        //~^ ERROR non-exhaustive patterns: `Color::Red` not covered
        Color::CustomRGBA { .. } => (),
        Color::Green => (),
    }
}

enum Direction {
    North,
    East,
    South,
    West,
}

fn enum_with_multiple_missing_variants() {
    match Direction::North {
        //~^ ERROR non-exhaustive patterns: `Direction::East`, `Direction::South` and `Direction::West` not covered
        Direction::North => (),
    }
}

enum ExcessiveEnum {
    First,
    Second,
    Third,
    Fourth,
    Fifth,
    Sixth,
    Seventh,
    Eighth,
    Ninth,
    Tenth,
    Eleventh,
    Twelfth,
}

fn enum_with_excessive_missing_variants() {
    match ExcessiveEnum::First {
        //~^ ERROR `ExcessiveEnum::Second`, `ExcessiveEnum::Third`, `ExcessiveEnum::Fourth` and 8 more not covered
        ExcessiveEnum::First => (),
    }
}

fn enum_struct_variant() {
    match Color::Red {
        //~^ ERROR non-exhaustive patterns: `Color::CustomRGBA { a: true, .. }` not covered
        Color::Red => (),
        Color::Green => (),
        Color::CustomRGBA { a: false, r: _, g: _, b: 0 } => (),
        Color::CustomRGBA { a: false, r: _, g: _, b: _ } => (),
    }
}

enum Enum {
    First,
    Second(bool),
}

fn vectors_with_nested_enums() {
    let x: &'static [Enum] = &[Enum::First, Enum::Second(false)];
    match *x {
        //~^ ERROR non-exhaustive patterns: `[Enum::Second(true), Enum::Second(false)]` not covered
        [] => (),
        [_] => (),
        [Enum::First, _] => (),
        [Enum::Second(true), Enum::First] => (),
        [Enum::Second(true), Enum::Second(true)] => (),
        [Enum::Second(false), _] => (),
        [_, _, ref tail @ .., _] => (),
    }
}

fn missing_nil() {
    match ((), false) {
        //~^ ERROR non-exhaustive patterns: `((), false)` not covered
        ((), true) => (),
    }
}

fn main() {}
