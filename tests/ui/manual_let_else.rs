#![allow(unused_braces, unused_variables, dead_code)]
#![allow(
    clippy::collapsible_else_if,
    clippy::unused_unit,
    clippy::let_unit_value,
    clippy::match_single_binding,
    clippy::never_loop,
    clippy::needless_if
)]
#![warn(clippy::manual_let_else)]

enum Variant {
    A(usize, usize),
    B(usize),
    C,
}

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

    // Tuples supported with multiple bindings
    let (w, S { v }) = if let (Some(v_some), w_some) = (g().map(|_| S { v: 0 }), 0) {
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

    fn e() -> Variant {
        Variant::A(0, 0)
    }

    let v = if let Variant::A(a, 0) = e() { a } else { return };

    // `mut v` is inserted into the pattern
    let mut v = if let Variant::B(b) = e() { b } else { return };

    // Nesting works
    let nested = Ok(Some(e()));
    let v = if let Ok(Some(Variant::B(b))) | Err(Some(Variant::A(b, _))) = nested {
        b
    } else {
        return;
    };
    // dot dot works
    let v = if let Variant::A(.., a) = e() { a } else { return };

    // () is preserved: a bit of an edge case but make sure it stays around
    let w = if let (Some(v), ()) = (g(), ()) { v } else { return };

    // Tuple structs work
    let w = if let Some(S { v: x }) = Some(S { v: 0 }) {
        x
    } else {
        return;
    };

    // Field init shorthand is suggested
    let v = if let Some(S { v: x }) = Some(S { v: 0 }) {
        x
    } else {
        return;
    };

    // Multi-field structs also work
    let (x, S { v }, w) = if let Some(U { v, w, x }) = None::<U<S<()>>> {
        (x, v, w)
    } else {
        return;
    };
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

    // Issue 10296
    // The let/else block in the else part is not divergent despite the presence of return
    let _x = if let Some(x) = Some(1) {
        x
    } else {
        let Some(_z) = Some(3) else {
            return
        };
        1
    };

    // This would require creation of a suggestion of the form
    // let v @ (Some(_), _) = (...) else { return };
    // Which is too advanced for our code, so we just bail.
    let v = if let (Some(v_some), w_some) = (g(), 0) {
        (w_some, v_some)
    } else {
        return;
    };
}

struct S<T> {
    v: T,
}

struct U<T> {
    v: T,
    w: T,
    x: T,
}
