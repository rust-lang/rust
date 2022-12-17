#![allow(unused_braces, unused_variables, dead_code)]
#![allow(clippy::collapsible_else_if, clippy::let_unit_value)]
#![warn(clippy::manual_let_else)]
// Ensure that we don't conflict with match -> if let lints
#![warn(clippy::single_match_else, clippy::single_match)]

fn f() -> Result<u32, u32> {
    Ok(0)
}

fn g() -> Option<()> {
    None
}

fn h() -> (Option<()>, Option<()>) {
    (None, None)
}

enum Variant {
    Foo,
    Bar(u32),
    Baz(u32),
}

fn build_enum() -> Variant {
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
        // More complex pattern for the identity arm and diverging arm
        let v = match h() {
            (Some(_), Some(_)) | (None, None) => continue,
            (Some(v), None) | (None, Some(v)) => v,
        };
        // Custom enums are supported as long as the "else" arm is a simple _
        let v = match build_enum() {
            _ => continue,
            Variant::Bar(v) | Variant::Baz(v) => v,
        };
    }

    // There is a _ in the diverging arm
    // TODO also support unused bindings aka _v
    let v = match f() {
        Ok(v) => v,
        Err(_) => return,
    };

    // Err(()) is an allowed pattern
    let v = match f().map_err(|_| ()) {
        Ok(v) => v,
        Err(()) => return,
    };

    let f = Variant::Bar(1);

    let _value = match f {
        Variant::Bar(_) | Variant::Baz(_) => (),
        _ => return,
    };
}

fn not_fire() {
    // Multiple diverging arms
    let v = match h() {
        _ => panic!(),
        (None, Some(_v)) => return,
        (Some(v), None) => v,
    };

    // Multiple identity arms
    let v = match h() {
        _ => panic!(),
        (None, Some(v)) => v,
        (Some(v), None) => v,
    };

    // No diverging arm at all, only identity arms.
    // This is no case for let else, but destructuring assignment.
    let v = match f() {
        Ok(v) => v,
        Err(e) => e,
    };

    // The identity arm has a guard
    let v = match g() {
        Some(v) if g().is_none() => v,
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

    // Custom enum where the diverging arm
    // explicitly mentions the variant
    let v = match build_enum() {
        Variant::Foo => return,
        Variant::Bar(v) | Variant::Baz(v) => v,
    };

    // The custom enum is surrounded by an Err()
    let v = match Err(build_enum()) {
        Ok(v) | Err(Variant::Bar(v) | Variant::Baz(v)) => v,
        Err(Variant::Foo) => return,
    };
}
