#![warn(clippy::shadow_same, clippy::shadow_reuse, clippy::shadow_unrelated)]

fn shadow_same() {
    let x = 1;
    let x = x;
    let mut x = &x;
    let x = &mut x;
    let x = *x;
}

fn shadow_reuse() -> Option<()> {
    let x = ([[0]], ());
    let x = x.0;
    let x = x[0];
    let [x] = x;
    let x = Some(x);
    let x = foo(x);
    let x = || x;
    let x = Some(1).map(|_| x)?;
    None
}

fn shadow_unrelated() {
    let x = 1;
    let x = 2;
}

fn syntax() {
    fn f(x: u32) {
        let x = 1;
    }
    let x = 1;
    match Some(1) {
        Some(1) => {},
        Some(x) => {
            let x = 1;
        },
        _ => {},
    }
    if let Some(x) = Some(1) {}
    while let Some(x) = Some(1) {}
    let _ = |[x]: [u32; 1]| {
        let x = 1;
    };
}

fn negative() {
    match Some(1) {
        Some(x) if x == 1 => {},
        Some(x) => {},
        None => {},
    }
    match [None, Some(1)] {
        [Some(x), None] | [None, Some(x)] => {},
        _ => {},
    }
    if let Some(x) = Some(1) {
        let y = 1;
    } else {
        let x = 1;
        let y = 1;
    }
    let x = 1;
    #[allow(clippy::shadow_unrelated)]
    let x = 1;
}

fn foo<T>(_: T) {}

fn question_mark() -> Option<()> {
    let val = 1;
    // `?` expands with a `val` binding
    None?;
    None
}

fn main() {}
