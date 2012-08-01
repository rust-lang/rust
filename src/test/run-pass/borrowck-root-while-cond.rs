fn borrow<T>(x: &T) -> &T {x}

fn main() {
    let rec = @{mut f: @22};
    while *borrow(rec.f) == 23 {}
}
