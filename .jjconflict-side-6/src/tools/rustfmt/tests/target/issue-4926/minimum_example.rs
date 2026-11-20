// rustfmt-struct_field_align_threshold: 30

struct X {
    a: i32,
    b: i32,
}

fn test(x: X) {
    let y = matches!(x, X { a: 1, .. });
}
