struct boxed_int {
    f: &int,
}

fn max(bi: &r/boxed_int, f: &r/int) -> int {
    if *bi.f > *f {*bi.f} else {*f}
}

fn with(bi: &boxed_int) -> int {
    let i = 22;
    max(bi, &i)
}

fn main() {
    let g = 21;
    let foo = boxed_int { f: &g };
    assert with(&foo) == 22;
}