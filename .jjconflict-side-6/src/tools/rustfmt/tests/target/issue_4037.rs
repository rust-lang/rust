fn f() {
    let x = 0;
    match x {
        0 => {}

        1 => {}
        _ => {} // foo
                // bar,
    }
}

fn wont_fmt() -> [(); 1] {
    [
        (), //
            // ,
    ]
}
