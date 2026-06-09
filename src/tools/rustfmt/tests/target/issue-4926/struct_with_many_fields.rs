// rustfmt-struct_field_align_threshold: 30

struct X {
    a: i32,
    b: i32,
    c: i32,
    d: i32,
    e: i32,
    f: i32,
    g: i32,
    h: i32,
    i: i32,
    j: i32,
    k: i32,
}

fn test(x: X) {
    let y = matches!(
        x,
        X {
            a: 1_000,
            b: 1_000,
            c: 1_000,
            d: 1_000,
            e: 1_000,
            f: 1_000,
            g: 1_000,
            h: 1_000,
            i: 1_000,
            j: 1_000,
            ..
        }
    );
}
