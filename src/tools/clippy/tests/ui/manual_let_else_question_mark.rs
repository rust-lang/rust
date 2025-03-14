#![allow(unused_braces, unused_variables, dead_code)]
#![allow(
    clippy::collapsible_else_if,
    clippy::unused_unit,
    clippy::let_unit_value,
    clippy::match_single_binding,
    clippy::never_loop
)]
#![warn(clippy::manual_let_else, clippy::question_mark)]

enum Variant {
    A(usize, usize),
    B(usize),
    C,
}

fn g() -> Option<(u8, u8)> {
    None
}

fn e() -> Variant {
    Variant::A(0, 0)
}

fn main() {}

fn foo() -> Option<()> {
    // Fire here, normal case
    let Some(v) = g() else { return None };
    //~^ question_mark

    // Don't fire here, the pattern is refutable
    let Variant::A(v, w) = e() else { return None };

    // Fire here, the pattern is irrefutable
    let Some((v, w)) = g() else { return None };
    //~^ question_mark

    // Don't fire manual_let_else in this instance: question mark can be used instead.
    let v = if let Some(v_some) = g() { v_some } else { return None };
    //~^ question_mark

    // Do fire manual_let_else in this instance: question mark cannot be used here due to the return
    // body.
    let v = if let Some(v_some) = g() {
        //~^ manual_let_else
        v_some
    } else {
        return Some(());
    };

    // Here we could also fire the question_mark lint, but we don't (as it's a match and not an if let).
    // So we still emit manual_let_else here. For the *resulting* code, we *do* emit the question_mark
    // lint, so for rustfix reasons, we allow the question_mark lint here.
    #[allow(clippy::question_mark)]
    {
        let v = match g() {
            //~^ manual_let_else
            Some(v_some) => v_some,
            _ => return None,
        };
    }

    // This is a copy of the case above where we'd fire the question_mark lint, but here we have allowed
    // it. Make sure that manual_let_else is fired as the fallback.
    #[allow(clippy::question_mark)]
    {
        let v = if let Some(v_some) = g() { v_some } else { return None };
        //~^ manual_let_else
    }

    Some(())
}

// lint not just `return None`, but also `return None;` (note the semicolon)
fn issue11993(y: Option<i32>) -> Option<i32> {
    let Some(x) = y else {
        return None;
    };
    //~^^^ question_mark

    // don't lint: more than one statement in the else body
    let Some(x) = y else {
        todo!();
        return None;
    };

    let Some(x) = y else {
        // Roses are red,
        // violets are blue,
        // please keep this comment,
        // it's art, you know?
        return None;
    };

    None
}
