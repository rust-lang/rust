#![allow(unused_braces, unused_variables, dead_code)]
#![allow(clippy::collapsible_else_if, clippy::let_unit_value)]
#![warn(clippy::manual_let_else)]
// Ensure that we don't conflict with match -> if let lints
#![warn(clippy::single_match_else, clippy::single_match)]

enum Variant {
    Foo,
    Bar(u32),
    Baz(u32),
}

fn f() -> Result<u32, u32> {
    Ok(0)
}

fn g() -> Option<()> {
    None
}

fn h() -> Variant {
    Variant::Foo
}

fn main() {}

fn fire() {
    let v = match g() {
        Some(v_some) => v_some,
        None => return,
    };

    let v = match g() {
        Some(v_some) => v_some,
        _ => return,
    };

    loop {
        // More complex pattern for the identity arm
        let v = match h() {
            Variant::Foo => continue,
            Variant::Bar(v) | Variant::Baz(v) => v,
        };
    }

    // There is a _ in the diverging arm
    // TODO also support unused bindings aka _v
    let v = match f() {
        Ok(v) => v,
        Err(_) => return,
    };
}

fn not_fire() {
    // Multiple diverging arms
    let v = match h() {
        Variant::Foo => panic!(),
        Variant::Bar(_v) => return,
        Variant::Baz(v) => v,
    };

    // Multiple identity arms
    let v = match h() {
        Variant::Foo => panic!(),
        Variant::Bar(v) => v,
        Variant::Baz(v) => v,
    };

    // No diverging arm at all, only identity arms.
    // This is no case for let else, but destructuring assignment.
    let v = match f() {
        Ok(v) => v,
        Err(e) => e,
    };

    // The identity arm has a guard
    let v = match h() {
        Variant::Bar(v) if g().is_none() => v,
        _ => return,
    };

    // The diverging arm has a guard
    let v = match f() {
        Err(v) if v > 0 => panic!(),
        Ok(v) | Err(v) => v,
    };

    // The diverging arm creates a binding
    let v = match f() {
        Ok(v) => v,
        Err(e) => panic!("error: {e}"),
    };
}
