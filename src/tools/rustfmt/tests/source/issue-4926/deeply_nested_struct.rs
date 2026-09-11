
// rustfmt-struct_field_align_threshold: 30

struct X {
    a: i32,
    b: i32,
    c: i32,
}

fn test(x: X) {
    let d = {
        let e = {
            let f = {
                let g = {
                    let h = {
                        let i = {
                            let j = {
                                matches!(
                                    x,
                                    X { a: 1_000, b: 1_000, .. }
                                )
                            };
                            j
                        };
                        i
                    };
                    h
                };
                g
            };
            f
        };
        e
    };
}