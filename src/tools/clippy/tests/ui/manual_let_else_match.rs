#![allow(unused_braces, unused_variables, dead_code)]
#![allow(
    clippy::collapsible_else_if,
    clippy::let_unit_value,
    clippy::redundant_at_rest_pattern
)]
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
        //~^ manual_let_else
        Some(v_some) => v_some,
        None => return,
    };

    let v = match g() {
        //~^ manual_let_else
        Some(v_some) => v_some,
        _ => return,
    };

    loop {
        // More complex pattern for the identity arm and diverging arm
        let v = match h() {
            //~^ manual_let_else
            (Some(v), None) | (None, Some(v)) => v,
            (Some(_), Some(_)) | (None, None) => continue,
        };
        // Custom enums are supported as long as the "else" arm is a simple _
        let v = match build_enum() {
            //~^ manual_let_else
            Variant::Bar(v) | Variant::Baz(v) => v,
            _ => continue,
        };
    }

    // There is a _ in the diverging arm
    // TODO also support unused bindings aka _v
    let v = match f() {
        //~^ manual_let_else
        Ok(v) => v,
        Err(_) => return,
    };

    // Err(()) is an allowed pattern
    let v = match f().map_err(|_| ()) {
        //~^ manual_let_else
        Ok(v) => v,
        Err(()) => return,
    };

    let f = Variant::Bar(1);

    let _value = match f {
        //~^ manual_let_else
        Variant::Bar(v) | Variant::Baz(v) => v,
        _ => return,
    };

    let _value = match Some(build_enum()) {
        //~^ manual_let_else
        Some(Variant::Bar(v) | Variant::Baz(v)) => v,
        _ => return,
    };

    let data = [1_u8, 2, 3, 4, 0, 0, 0, 0];
    let data = match data.as_slice() {
        //~^ manual_let_else
        [data @ .., 0, 0, 0, 0] | [data @ .., 0, 0] | [data @ .., 0] => data,
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

    // Issue 10241
    // The non-divergent arm arrives in second position and
    // may cover values already matched in the first arm.
    let v = match h() {
        (Some(_), Some(_)) | (None, None) => return,
        (Some(v), _) | (None, Some(v)) => v,
    };

    let v = match build_enum() {
        _ => return,
        Variant::Bar(v) | Variant::Baz(v) => v,
    };

    let data = [1_u8, 2, 3, 4, 0, 0, 0, 0];
    let data = match data.as_slice() {
        [] | [0, 0] => return,
        [data @ .., 0, 0, 0, 0] | [data @ .., 0, 0] | [data @ ..] => data,
    };
}

fn issue11579() {
    let msg = match Some("hi") {
        //~^ manual_let_else
        Some(m) => m,
        _ => unreachable!("can't happen"),
    };
}

#[derive(Clone, Copy)]
struct Issue9939<T> {
    avalanche: T,
}

fn issue9939() {
    let issue = Some(Issue9939 { avalanche: 1 });
    let tornado = match issue {
        //~^ manual_let_else
        Some(Issue9939 { avalanche }) => avalanche,
        _ => unreachable!("can't happen"),
    };
    let issue = Some(Issue9939 { avalanche: true });
    let acid_rain = match issue {
        //~^ manual_let_else
        Some(Issue9939 { avalanche: tornado }) => tornado,
        _ => unreachable!("can't happen"),
    };
    assert_eq!(tornado, 1);
    assert!(acid_rain);

    // without shadowing
    let _y = match issue {
        //~^ manual_let_else
        _x @ Some(Issue9939 { avalanche }) => avalanche,
        None => unreachable!("can't happen"),
    };

    // with shadowing
    let _x = match issue {
        //~^ manual_let_else
        _x @ Some(Issue9939 { avalanche }) => avalanche,
        None => unreachable!("can't happen"),
    };
}

#[derive(Clone, Copy)]
struct Issue9939b<T, U> {
    earthquake: T,
    hurricane: U,
}

fn issue9939b() {
    let issue = Some(Issue9939b {
        earthquake: true,
        hurricane: 1,
    });
    let (issue, drought, flood) = match issue {
        //~^ manual_let_else
        flood @ Some(Issue9939b { earthquake, hurricane }) => (flood, hurricane, earthquake),
        None => unreachable!("can't happen"),
    };
    assert_eq!(drought, 1);
    assert!(flood);
    assert!(issue.is_some());

    // without shadowing
    let (_y, erosion) = match issue {
        //~^ manual_let_else
        _x @ Some(Issue9939b { earthquake, hurricane }) => (hurricane, earthquake),
        None => unreachable!("can't happen"),
    };
    assert!(erosion);

    // with shadowing
    let (_x, erosion) = match issue {
        //~^ manual_let_else
        _x @ Some(Issue9939b { earthquake, hurricane }) => (hurricane, earthquake),
        None => unreachable!("can't happen"),
    };
    assert!(erosion);
}
