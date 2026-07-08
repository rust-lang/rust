//@ check-pass

#![feature(stmt_expr_attributes)]
#![expect(
    clippy::never_loop,
    clippy::redundant_pattern_matching,
    clippy::while_immutable_condition
)]

fn main() {
    #[clippy::author]
    for y in 0..10 {
        let z = y;
    }

    #[clippy::author]
    for _ in 0..10 {
        break;
    }

    #[clippy::author]
    'label: for _ in 0..10 {
        break 'label;
    }

    let a = true;

    #[clippy::author]
    while a {
        break;
    }

    #[clippy::author]
    while let true = a {
        break;
    }

    #[clippy::author]
    loop {
        break;
    }
}
