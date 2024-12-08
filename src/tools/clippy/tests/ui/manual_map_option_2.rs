#![warn(clippy::manual_map)]
#![allow(clippy::toplevel_ref_arg)]

fn main() {
    // Lint. `y` is declared within the arm, so it isn't captured by the map closure
    let _ = match Some(0) {
        Some(x) => Some({
            let y = (String::new(), String::new());
            (x, y.0)
        }),
        None => None,
    };

    // Don't lint. `s` is borrowed until partway through the arm, but needs to be captured by the map
    // closure
    let s = Some(String::new());
    let _ = match &s {
        Some(x) => Some((x.clone(), s)),
        None => None,
    };

    // Don't lint. `s` is borrowed until partway through the arm, but needs to be captured by the map
    // closure
    let s = Some(String::new());
    let _ = match &s {
        Some(x) => Some({
            let clone = x.clone();
            let s = || s;
            (clone, s())
        }),
        None => None,
    };

    // Don't lint. `s` is borrowed until partway through the arm, but needs to be captured as a mutable
    // reference by the map closure
    let mut s = Some(String::new());
    let _ = match &s {
        Some(x) => Some({
            let clone = x.clone();
            let ref mut s = s;
            (clone, s)
        }),
        None => None,
    };

    // Lint. `s` is captured by reference, so no lifetime issues.
    let s = Some(String::new());
    let _ = match &s {
        Some(x) => Some({ if let Some(ref s) = s { (x.clone(), s) } else { panic!() } }),
        None => None,
    };

    // Issue #7820
    unsafe fn f(x: u32) -> u32 {
        x
    }
    unsafe {
        let _ = match Some(0) {
            Some(x) => Some(f(x)),
            None => None,
        };
    }
    let _ = match Some(0) {
        Some(x) => unsafe { Some(f(x)) },
        None => None,
    };
    let _ = match Some(0) {
        Some(x) => Some(unsafe { f(x) }),
        None => None,
    };
}
