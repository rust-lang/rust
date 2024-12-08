// This is a test for the new temporary lifetime behaviour as implemented for RFC 3606.
// In essence, with #3606 we can write the following variable initialisation without
// a borrow checking error because the temporary lifetime is automatically extended.
// ```rust
// let x = if condition() {
//    &something()
// } else {
//    &something_else()
// };
// ```
// More details can be found in https://github.com/rust-lang/rfcs/pull/3606

//@ run-pass
//@ check-run-results
//@ revisions: edition2021 edition2024
//@ [edition2021] edition: 2021
//@ [edition2024] edition: 2024

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
