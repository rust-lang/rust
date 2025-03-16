// Check shadowing in if let guards works as expected.
//@ check-pass
//@ edition: 2024

#![feature(if_let_guard)]

fn main() {
    let x: Option<Option<i32>> = Some(Some(6));
    match x {
        Some(x) if let Some(x) = x => {
            let _: i32 = x;
        }
        _ => {}
    }

    let y: Option<Option<Option<i32>>> = Some(Some(Some(-24)));
    match y {
        Some(y) if let Some(y) = y && let Some(y) = y => {
            let _: i32 = y;
        }
        _ => {}
    }
}
