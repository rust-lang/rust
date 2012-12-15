struct S<T>(T);

fn main() {
    let s = S(2);
    io::println(s.to_str());
}

