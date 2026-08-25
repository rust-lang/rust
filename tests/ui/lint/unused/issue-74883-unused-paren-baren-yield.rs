#![feature(coroutine_trait)]
#![feature(coroutines, stmt_expr_attributes)]
#![deny(unused_braces, unused_parens)]

use std::ops::Coroutine;
use std::pin::Pin;

fn main() {
    let mut x = #[coroutine] |_| {
        while let Some(_) = (yield) {}
        while let Some(_) = {yield} {}

        // Only warn these cases
        while let Some(_) = ({yield}) {} //~ ERROR: unnecessary parentheses
        while let Some(_) = ((yield)) {} //~ ERROR: unnecessary parentheses
        {{yield}}; //~ ERROR: unnecessary braces
        {( yield )}; //~ ERROR: unnecessary parentheses
        while let Some(_) = {(yield)} {} //~ ERROR: unnecessary parentheses
        while let Some(_) = {{yield}} {} //~ ERROR: unnecessary braces

        // FIXME: It'd be great if we could also warn them.
        ((yield));
        ({ yield });
    };
    let _ = Pin::new(&mut x).resume(Some(5));
}
