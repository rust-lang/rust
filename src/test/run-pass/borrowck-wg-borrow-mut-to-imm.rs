fn g(x: &Option<int>) {
    io::println(x.get().to_str());
}

fn f(x: &mut Option<int>) {
    g(&*x);
}

fn main() {
    let mut x = ~Some(3);
    f(x);
}
