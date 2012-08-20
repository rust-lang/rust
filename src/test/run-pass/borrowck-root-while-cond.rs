fn borrow<T>(x: &r/T) -> &r/T {x}

fn main() {
    let rec = @{mut f: @22};
    while *borrow(rec.f) == 23 {}
}
