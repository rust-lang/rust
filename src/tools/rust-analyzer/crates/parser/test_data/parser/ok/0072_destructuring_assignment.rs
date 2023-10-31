fn foo() {
    let (mut a, mut b) = (0, 1);
    (b, a, ..) = (a, b);
    (_) = ..;
    struct S { a: i32 }
    S { .. } = S { ..S::default() };
    Some(..) = Some(0);
    Ok(_) = 0;
    let (a, b);
    [a, .., b] = [1, .., 2];
    (_, _) = (a, b);
    (_) = (a, b);
    _ = (a, b);
}
