// build-pass

#![feature(inline_const_pat, exclusive_range_pattern)]

fn main() {
    const N: u32 = 10;
    let x: u32 = 3;

    match x {
        1 ..= const { N + 1 } => {},
        _ => {},
    }

    match x {
        const { N - 1 } ..= 10 => {},
        _ => {},
    }

    match x {
        const { N - 1 } ..= const { N + 1 } => {},
        _ => {},
    }

    match x {
        .. const { N + 1 } => {},
        _ => {},
    }

    match x {
        const { N - 1 } .. => {},
        _ => {},
    }

    match x {
        ..= const { N + 1 } => {},
        _ => {}
    }
}
