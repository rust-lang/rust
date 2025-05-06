#![feature(try_blocks)]
#![allow(unused_braces, unused_variables, dead_code)]
#![allow(
    clippy::collapsible_else_if,
    clippy::unused_unit,
    clippy::let_unit_value,
    clippy::match_single_binding,
    clippy::never_loop,
    clippy::needless_if,
    clippy::diverging_sub_expression,
    clippy::single_match,
    clippy::manual_unwrap_or_default
)]
#![warn(clippy::manual_let_else)]
//@no-rustfix
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
    //~^ manual_let_else

    let v = if let Some(v_some) = g() {
        //~^ manual_let_else

        v_some
    } else {
        return;
    };

    let v = if let Some(v) = g() {
        //~^ manual_let_else

        // Blocks around the identity should have no impact
        { { v } }
    } else {
        // Some computation should still make it fire
        g();
        return;
    };

    // continue and break diverge
    loop {
        let v = if let Some(v_some) = g() { v_some } else { continue };
        //~^ manual_let_else

        let v = if let Some(v_some) = g() { v_some } else { break };
        //~^ manual_let_else
    }

    // panic also diverges
    let v = if let Some(v_some) = g() { v_some } else { panic!() };
    //~^ manual_let_else

    // abort also diverges
    let v = if let Some(v_some) = g() {
        //~^ manual_let_else

        v_some
    } else {
        std::process::abort()
    };

    // If whose two branches diverge also diverges
    let v = if let Some(v_some) = g() {
        //~^ manual_let_else

        v_some
    } else {
        if true { return } else { panic!() }
    };

    // Diverging after an if still makes the block diverge:
    let v = if let Some(v_some) = g() {
        //~^ manual_let_else

        v_some
    } else {
        if true {}
        panic!();
    };

    // The final expression will need to be turned into a statement.
    let v = if let Some(v_some) = g() {
        //~^ manual_let_else

        v_some
    } else {
        panic!();
        ()
    };

    // Even if the result is buried multiple expressions deep.
    let v = if let Some(v_some) = g() {
        //~^ manual_let_else

        v_some
    } else {
        panic!();
        if true {
            match 0 {
                0 => (),
                _ => (),
            }
        } else {
            panic!()
        }
    };

    // Or if a break gives the value.
    let v = if let Some(v_some) = g() {
        //~^ manual_let_else

        v_some
    } else {
        loop {
            panic!();
            break ();
        }
    };

    // Even if the break is in a weird position.
    let v = if let Some(v_some) = g() {
        //~^ manual_let_else

        v_some
    } else {
        'a: loop {
            panic!();
            loop {
                match 0 {
                    0 if (return break 'a ()) => {},
                    _ => {},
                }
            }
        }
    };

    // A match diverges if all branches diverge:
    let v = if let Some(v_some) = g() {
        //~^ manual_let_else

        v_some
    } else {
        match 0 {
            0 if true => panic!(),
            _ => panic!(),
        };
    };

    // An if's expression can cause divergence:
    let v = if let Some(v_some) = g() {
        //~^ manual_let_else

        v_some
    } else {
        if panic!() {};
    };

    // An expression of a match can cause divergence:
    let v = if let Some(v_some) = g() {
        //~^ manual_let_else

        v_some
    } else {
        match panic!() {
            _ => {},
        };
    };

    // Top level else if
    let v = if let Some(v_some) = g() {
        //~^ manual_let_else

        v_some
    } else if true {
        return;
    } else {
        panic!("diverge");
    };

    // All match arms diverge
    let v = if let Some(v_some) = g() {
        //~^ manual_let_else

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
        //~^ manual_let_else

        v_some
    } else {
        return;
    };

    // Tuples supported with multiple bindings
    let (w, S { v }) = if let (Some(v_some), w_some) = (g().map(|_| S { v: 0 }), 0) {
        //~^ manual_let_else

        (w_some, v_some)
    } else {
        return;
    };

    // entirely inside macro lints
    macro_rules! create_binding_if_some {
        ($n:ident, $e:expr) => {
            let $n = if let Some(v) = $e { v } else { return };
            //~^ manual_let_else
        };
    }
    create_binding_if_some!(w, g());

    fn e() -> Variant {
        Variant::A(0, 0)
    }

    let v = if let Variant::A(a, 0) = e() { a } else { return };
    //~^ manual_let_else

    // `mut v` is inserted into the pattern
    let mut v = if let Variant::B(b) = e() { b } else { return };
    //~^ manual_let_else

    // Nesting works
    let nested = Ok(Some(e()));
    let v = if let Ok(Some(Variant::B(b))) | Err(Some(Variant::A(b, _))) = nested {
        //~^ manual_let_else

        b
    } else {
        return;
    };
    // dot dot works
    let v = if let Variant::A(.., a) = e() { a } else { return };
    //~^ manual_let_else

    // () is preserved: a bit of an edge case but make sure it stays around
    let w = if let (Some(v), ()) = (g(), ()) { v } else { return };
    //~^ manual_let_else

    // Tuple structs work
    let w = if let Some(S { v: x }) = Some(S { v: 0 }) {
        //~^ manual_let_else

        x
    } else {
        return;
    };

    // Field init shorthand is suggested
    let v = if let Some(S { v: x }) = Some(S { v: 0 }) {
        //~^ manual_let_else

        x
    } else {
        return;
    };

    // Multi-field structs also work
    let (x, S { v }, w) = if let Some(U { v, w, x }) = None::<U<S<()>>> {
        //~^ manual_let_else

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
    let Some(a) = (if let Some(b) = Some(Some(())) { b } else { return }) else {
        panic!()
    };

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
        //~^ manual_let_else
        Some(value) => value,
        _ => macro_call!(),
    };

    // Issue 10296
    // The let/else block in the else part is not divergent despite the presence of return
    let _x = if let Some(x) = Some(1) {
        x
    } else {
        let Some(_z) = Some(3) else { return };
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

    // A break that skips the divergent statement will cause the expression to be non-divergent.
    let _x = if let Some(x) = Some(0) {
        x
    } else {
        'foo: loop {
            break 'foo 0;
            panic!();
        }
    };

    // Even in inner loops.
    let _x = if let Some(x) = Some(0) {
        x
    } else {
        'foo: {
            loop {
                break 'foo 0;
            }
            panic!();
        }
    };

    // But a break that can't ever be reached still affects divergence checking.
    let _x = if let Some(x) = g() {
        x
    } else {
        'foo: {
            'bar: loop {
                loop {
                    break 'bar ();
                }
                break 'foo ();
            }
            panic!();
        };
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

fn issue12337() {
    // We want to generally silence question_mark lints within try blocks, since `?` has different
    // behavior to `return`, and question_mark calls into manual_let_else logic, so make sure that
    // we still emit a lint for manual_let_else
    let _: Option<()> = try {
        let v = if let Some(v_some) = g() { v_some } else { return };
        //~^ manual_let_else
    };
}

mod issue13768 {
    enum Foo {
        Str(String),
        None,
    }

    fn foo(value: Foo) {
        let signature = match value {
            //~^ manual_let_else
            Foo::Str(ref val) => val,
            _ => {
                println!("No signature found");
                return;
            },
        };
    }

    enum Bar {
        Str { inner: String },
        None,
    }

    fn bar(mut value: Bar) {
        let signature = match value {
            //~^ manual_let_else
            Bar::Str { ref mut inner } => inner,
            _ => {
                println!("No signature found");
                return;
            },
        };
    }
}

mod issue14598 {
    fn bar() -> Result<bool, &'static str> {
        let value = match foo() {
            //~^ manual_let_else
            Err(_) => return Err("abc"),
            Ok(value) => value,
        };

        let w = Some(0);
        let v = match w {
            //~^ manual_let_else
            None => return Err("abc"),
            Some(x) => x,
        };

        enum Foo<T> {
            Foo(T),
        }

        let v = match Foo::Foo(Some(())) {
            Foo::Foo(Some(_)) => return Err("abc"),
            Foo::Foo(v) => v,
        };

        Ok(value == 42)
    }

    fn foo() -> Result<u32, &'static str> {
        todo!()
    }
}
