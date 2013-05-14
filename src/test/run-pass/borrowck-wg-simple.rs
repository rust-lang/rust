fn f(x: &int) {
    io::println(x.to_str());
}

pub fn main() {
    let x = @mut 3;
    f(x);
}
