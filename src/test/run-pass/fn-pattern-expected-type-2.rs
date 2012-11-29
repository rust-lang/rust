fn main() {
    let v = [ (1, 2), (3, 4), (5, 6) ];
    for v.each |&(x, y)| {
        io::println(y.to_str());
        io::println(x.to_str());
    }
}

