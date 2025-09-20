// Regression test for #146586

//@ only-linux
//@ compile-flags: --error-format=human --color=always

enum Enum {
    Unit,
    Tuple(i32),
    Struct { x: i32 },
}

fn foo(_: Enum) {}

fn main() {
    foo(Enum::Unit);
    foo(Enum::Tuple);
    foo(Enum::Struct); // Suggestion was malformed
    foo(Enum::Unit());
    foo(Enum::Tuple());
    foo(Enum::Struct());
    foo(Enum::Unit {});
    foo(Enum::Tuple {});
    foo(Enum::Struct {});
    foo(Enum::Unit(0));
    foo(Enum::Tuple(0));
    foo(Enum::Struct(0));
    foo(Enum::Unit { x: 0 });
    foo(Enum::Tuple { x: 0 });
    foo(Enum::Struct { x: 0 });
    foo(Enum::Unit(0, 0));
    foo(Enum::Tuple(0, 0));
    foo(Enum::Struct(0, 0));
    foo(Enum::Unit { x: 0, y: 0 });
    foo(Enum::Tuple { x: 0, y: 0 });
    foo(Enum::Struct { x: 0, y: 0 });
    foo(Enum::unit); // Suggestion was malformed
    foo(Enum::tuple); // Suggestion is enhanced
    foo(Enum::r#struct); // Suggestion was malformed
    foo(Enum::unit());
    foo(Enum::tuple());
    foo(Enum::r#struct());
    foo(Enum::unit {}); // Suggestion could be enhanced
    foo(Enum::tuple {}); // Suggestion could be enhanced
    foo(Enum::r#struct {}); // Suggestion could be enhanced
    foo(Enum::unit(0));
    foo(Enum::tuple(0));
    foo(Enum::r#struct(0));
    foo(Enum::unit { x: 0 }); // Suggestion could be enhanced
    foo(Enum::tuple { x: 0 }); // Suggestion could be enhanced
    foo(Enum::r#struct { x: 0 });
    foo(Enum::unit(0, 0));
    foo(Enum::tuple(0, 0));
    foo(Enum::r#struct(0, 0));
    foo(Enum::unit { x: 0, y: 0 }); // Suggestion could be enhanced
    foo(Enum::tuple { x: 0, y: 0 }); // Suggestion could be enhanced
    foo(Enum::r#struct { x: 0, y: 0 }); // Suggestion could be enhanced
}
