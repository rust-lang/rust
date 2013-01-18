fn f(x: &int) {
    io::println(x.to_str());
}

fn main() {
    let x = @mut 3;
    f(x);
}

