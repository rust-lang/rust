struct S<T>(T);

fn main() {
    let s = S(2i);
    io::println(s.to_str());
}

