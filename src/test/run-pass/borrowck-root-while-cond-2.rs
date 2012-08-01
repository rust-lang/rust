fn main() {
    let rec = @{mut f: @{g: ~[1, 2, 3]}};
    while rec.f.g.len() == 23 {}
}
