//@ run-pass
//@ revisions: edition2021 edition2024
//@ [edition2021] edition: 2021
//@ [edition2024] edition: 2024
//@ [edition2024] compile-flags: -Z unstable-options

#![cfg_attr(edition2024, feature(shorter_tail_lifetimes))]

fn temp() -> (String, i32) {
    (String::from("Hello"), 1)
}

fn main() {
    let a = &temp();
    let b = [(&temp(),)];
    let c = &temp().0;
    let d = &temp().0[..];
    let e = {
        let _ = 123;
        &(*temp().0)[..]
    };
    let f = if true { &temp() } else { &temp() };
    let g = match true {
        true => &temp(),
        false => {
            let _ = 123;
            &temp()
        }
    };
    let h = match temp() {
        // The {} moves the value, making a new temporary.
        owned_non_temporary => &{ owned_non_temporary },
    };
    println!("{a:?} {b:?} {c:?} {d:?} {e:?} {f:?} {g:?} {h:?}");
}
