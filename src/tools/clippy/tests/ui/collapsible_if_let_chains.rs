//@revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024
//@[edition2021] check-pass

#![warn(clippy::collapsible_if)]

fn main() {
    if let Some(a) = Some(3) {
        // with comment, so do not lint
        if let Some(b) = Some(4) {
            let _ = a + b;
        }
    }

    //~[edition2024]v collapsible_if
    if let Some(a) = Some(3) {
        if let Some(b) = Some(4) {
            let _ = a + b;
        }
    }

    //~[edition2024]v collapsible_if
    if let Some(a) = Some(3) {
        if a + 1 == 4 {
            let _ = a;
        }
    }

    //~[edition2024]v collapsible_if
    if Some(3) == Some(4).map(|x| x - 1) {
        if let Some(b) = Some(4) {
            let _ = b;
        }
    }

    fn truth() -> bool {
        true
    }

    // Prefix:
    //~[edition2024]v collapsible_if
    if let 0 = 1 {
        if truth() {}
    }

    // Suffix:
    //~[edition2024]v collapsible_if
    if truth() {
        if let 0 = 1 {}
    }

    // Midfix:
    //~[edition2024]vvv collapsible_if
    //~[edition2024]v collapsible_if
    if truth() {
        if let 0 = 1 {
            if truth() {}
        }
    }
}

#[clippy::msrv = "1.87.0"]
fn msrv_1_87() {
    if let 0 = 1 {
        if true {}
    }
}

#[clippy::msrv = "1.88.0"]
fn msrv_1_88() {
    //~[edition2024]v collapsible_if
    if let 0 = 1 {
        if true {}
    }
}
