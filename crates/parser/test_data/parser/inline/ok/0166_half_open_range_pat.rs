fn f() {
    let 0 .. = 1u32;
    let 0..: _ = 1u32;

    match 42 {
        0 .. if true => (),
        _ => (),
    }
}
