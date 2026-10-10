//@ only-linux
//@ compile-flags: --error-format=human --color=always

enum Enum {
    Unit,
    Tuple(i32),
    Struct { x: i32 },
}

fn main() {
    Enum::Unit;
    Enum::Tuple;
    Enum::Struct;
    Enum::Unit();
    Enum::Tuple();
    Enum::Struct();
    Enum::Unit {};
    Enum::Tuple {};
    Enum::Struct {};
    Enum::Unit(0);
    Enum::Tuple(0);
    Enum::Struct(0);
    Enum::Unit { x: 0 };
    Enum::Tuple { x: 0 };
    Enum::Struct { x: 0 }; // ok
    Enum::Unit(0, 0);
    Enum::Tuple(0, 0);
    Enum::Struct(0, 0);
    Enum::Unit { x: 0, y: 0 };

    Enum::Tuple { x: 0, y: 0 };

    Enum::Struct { x: 0, y: 0 };
    Enum::unit;
    Enum::tuple;
    Enum::r#struct;
    Enum::unit();
    Enum::tuple();
    Enum::r#struct();
    Enum::unit {};
    Enum::tuple {};
    Enum::r#struct {};
    Enum::unit(0);
    Enum::tuple(0);
    Enum::r#struct(0);
    Enum::unit { x: 0 };
    Enum::tuple { x: 0 };
    Enum::r#struct { x: 0 };
    Enum::unit(0, 0);
    Enum::tuple(0, 0);
    Enum::r#struct(0, 0);
    Enum::unit { x: 0, y: 0 };
    Enum::tuple { x: 0, y: 0 };
    Enum::r#struct { x: 0, y: 0 };
}
