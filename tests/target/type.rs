fn types() {
    let x: [Vec<_>] = [];
    let y: *mut [SomeType; konst_funk()] = expr();
    let z: (// #digits
            usize,
            // exp
            i16) = funk();
    let z: (usize /* #digits */, i16 /* exp */) = funk();
}
