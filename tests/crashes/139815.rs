//@ known-bug: #139815

#![feature(generic_const_exprs)]
fn is_123<const N: usize>(
    x: [u32; {
        N + 1;
        5
    }],
) -> bool {
    match x {
        [1, 2] => true,
        _ => false,
    }
}
