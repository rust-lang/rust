// rustfmt-struct_field_align_threshold: 30

struct X {
    really_really_long_field_a: i32,
    really_really_really_long_field_b: i32,
    really_really_really_really_long_field_c: i32,
    really_really_really_really_really_long_field_d: i32,
    really_really_really_really_really_really_long_field_e: i32,
    f: i32,
}

fn test(x: X) {
    let y = matches!(x, X {
        really_really_long_field_a: 10,
        really_really_really_long_field_b: 10,
        really_really_really_really_long_field_c: 10,
        really_really_really_really_really_long_field_d: 10,
        really_really_really_really_really_really_long_field_e: 10,
        ..
    });
}
