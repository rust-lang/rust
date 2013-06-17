struct S<T>(T);

pub fn main() {
    let s = S(2i);
    println(s.to_str());
}
