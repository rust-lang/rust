// Check shadowing in if let guards works as expected.

//@ check-pass
//@revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024

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
