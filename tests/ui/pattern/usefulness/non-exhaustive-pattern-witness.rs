struct Foo {
    first: bool,
    second: Option<[usize; 4]>
}

fn struct_with_a_nested_enum_and_vector() {
    match (Foo { first: true, second: None }) {
//~^ ERROR match is non-exhaustive
        Foo { first: true, second: None } => (),
        Foo { first: true, second: Some(_) } => (),
        Foo { first: false, second: None } => (),
        Foo { first: false, second: Some([1, 2, 3, 4]) } => ()
    }
}

enum Color {
    Red,
    Green,
    CustomRGBA { a: bool, r: u8, g: u8, b: u8 }
}

fn enum_with_single_missing_variant() {
    match Color::Red {
    //~^ ERROR match is non-exhaustive
        Color::CustomRGBA { .. } => (),
        Color::Green => ()
    }
}

enum Direction {
    North, East, South, West
}

fn enum_with_multiple_missing_variants() {
    match Direction::North {
    //~^ ERROR match is non-exhaustive
        Direction::North => ()
    }
}

enum ExcessiveEnum {
    First, Second, Third, Fourth, Fifth, Sixth, Seventh, Eighth, Ninth, Tenth, Eleventh, Twelfth
}

fn enum_with_excessive_missing_variants() {
    match ExcessiveEnum::First {
    //~^ ERROR match is non-exhaustive [E0004]
        ExcessiveEnum::First => ()
    }
}

fn enum_struct_variant() {
    match Color::Red {
    //~^ ERROR match is non-exhaustive
        Color::Red => (),
        Color::Green => (),
        Color::CustomRGBA { a: false, r: _, g: _, b: 0 } => (),
        Color::CustomRGBA { a: false, r: _, g: _, b: _ } => ()
    }
}

enum Enum {
    First,
    Second(bool)
}

fn vectors_with_nested_enums() {
    let x: &'static [Enum] = &[Enum::First, Enum::Second(false)];
    match *x {
    //~^ ERROR match is non-exhaustive
        [] => (),
        [_] => (),
        [Enum::First, _] => (),
        [Enum::Second(true), Enum::First] => (),
        [Enum::Second(true), Enum::Second(true)] => (),
        [Enum::Second(false), _] => (),
        [_, _, ref tail @ .., _] => ()
    }
}

fn missing_nil() {
    match ((), false) {
    //~^ ERROR match is non-exhaustive
        ((), true) => ()
    }
}

fn main() {}
