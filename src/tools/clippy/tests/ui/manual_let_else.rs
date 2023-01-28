#![allow(unused_braces, unused_variables, dead_code)]
#![allow(
    clippy::collapsible_else_if,
    clippy::unused_unit,
    clippy::let_unit_value,
    clippy::match_single_binding,
    clippy::never_loop
)]
#![warn(clippy::manual_let_else)]

fn g() -> Option<()> {
    None
}

fn main() {}

fn fire() {
    let v = if let Some(v_some) = g() { v_some } else { return };
    let v = if let Some(v_some) = g() {
        v_some
    } else {
        return;
    };

    let v = if let Some(v) = g() {
        // Blocks around the identity should have no impact
        {
            { v }
        }
    } else {
        // Some computation should still make it fire
        g();
        return;
    };

    // continue and break diverge
    loop {
        let v = if let Some(v_some) = g() { v_some } else { continue };
        let v = if let Some(v_some) = g() { v_some } else { break };
    }

    // panic also diverges
    let v = if let Some(v_some) = g() { v_some } else { panic!() };

    // abort also diverges
    let v = if let Some(v_some) = g() {
        v_some
    } else {
        std::process::abort()
    };

    // If whose two branches diverge also diverges
    let v = if let Some(v_some) = g() {
        v_some
    } else {
        if true { return } else { panic!() }
    };

    // Diverging after an if still makes the block diverge:
    let v = if let Some(v_some) = g() {
        v_some
    } else {
        if true {}
        panic!();
    };

    // A match diverges if all branches diverge:
    // Note: the corresponding let-else requires a ; at the end of the match
    // as otherwise the type checker does not turn it into a ! type.
    let v = if let Some(v_some) = g() {
        v_some
    } else {
        match () {
            _ if panic!() => {},
            _ => panic!(),
        }
    };

    // An if's expression can cause divergence:
    let v = if let Some(v_some) = g() { v_some } else { if panic!() {} };

    // An expression of a match can cause divergence:
    let v = if let Some(v_some) = g() {
        v_some
    } else {
        match panic!() {
            _ => {},
        }
    };

    // Top level else if
    let v = if let Some(v_some) = g() {
        v_some
    } else if true {
        return;
    } else {
        panic!("diverge");
    };

    // All match arms diverge
    let v = if let Some(v_some) = g() {
        v_some
    } else {
        match (g(), g()) {
            (Some(_), None) => return,
            (None, Some(_)) => {
                if true {
                    return;
                } else {
                    panic!();
                }
            },
            _ => return,
        }
    };

    // Tuples supported for the declared variables
    let (v, w) = if let Some(v_some) = g().map(|v| (v, 42)) {
        v_some
    } else {
        return;
    };

    // Tuples supported for the identity block and pattern
    let v = if let (Some(v_some), w_some) = (g(), 0) {
        (w_some, v_some)
    } else {
        return;
    };

    // entirely inside macro lints
    macro_rules! create_binding_if_some {
        ($n:ident, $e:expr) => {
            let $n = if let Some(v) = $e { v } else { return };
        };
    }
    create_binding_if_some!(w, g());
}

fn not_fire() {
    let v = if let Some(v_some) = g() {
        // Nothing returned. Should not fire.
    } else {
        return;
    };

    let w = 0;
    let v = if let Some(v_some) = g() {
        // Different variable than v_some. Should not fire.
        w
    } else {
        return;
    };

    let v = if let Some(v_some) = g() {
        // Computation in then clause. Should not fire.
        g();
        v_some
    } else {
        return;
    };

    let v = if let Some(v_some) = g() {
        v_some
    } else {
        if false {
            return;
        }
        // This doesn't diverge. Should not fire.
        ()
    };

    let v = if let Some(v_some) = g() {
        v_some
    } else {
        // There is one match arm that doesn't diverge. Should not fire.
        match (g(), g()) {
            (Some(_), None) => return,
            (None, Some(_)) => return,
            (Some(_), Some(_)) => (),
            _ => return,
        }
    };

    let v = if let Some(v_some) = g() {
        v_some
    } else {
        // loop with a break statement inside does not diverge.
        loop {
            break;
        }
    };

    enum Uninhabited {}
    fn un() -> Uninhabited {
        panic!()
    }
    let v = if let Some(v_some) = None {
        v_some
    } else {
        // Don't lint if the type is uninhabited but not !
        un()
    };

    fn question_mark() -> Option<()> {
        let v = if let Some(v) = g() {
            v
        } else {
            // Question mark does not diverge
            g()?
        };
        Some(v)
    }

    // Macro boundary inside let
    macro_rules! some_or_return {
        ($e:expr) => {
            if let Some(v) = $e { v } else { return }
        };
    }
    let v = some_or_return!(g());

    // Also macro boundary inside let, but inside a macro
    macro_rules! create_binding_if_some_nf {
        ($n:ident, $e:expr) => {
            let $n = some_or_return!($e);
        };
    }
    create_binding_if_some_nf!(v, g());

    // Already a let-else
    let Some(a) = (if let Some(b) = Some(Some(())) { b } else { return }) else { panic!() };

    // If a type annotation is present, don't lint as
    // expressing the type might be too hard
    let v: () = if let Some(v_some) = g() { v_some } else { panic!() };

    // Issue 9940
    // Suggestion should not expand macros
    macro_rules! macro_call {
        () => {
            return ()
        };
    }

    let ff = Some(1);
    let _ = match ff {
        Some(value) => value,
        _ => macro_call!(),
    };
}
