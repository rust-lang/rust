struct S<T>(T);

pub fn main() {
    let s = S(2i);
    io::println(s.to_str());
}
