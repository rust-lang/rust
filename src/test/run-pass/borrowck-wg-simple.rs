fn f(x: &int) {
    println(x.to_str());
}

pub fn main() {
    let x = @mut 3;
    f(x);
}
