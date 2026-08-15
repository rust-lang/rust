fn f() {
    let _ = match Some(1) {
        Some(_) => 1,
        None => 2,
    } else {
        return
    };
}