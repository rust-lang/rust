fn g(x: &Option<int>) {
    println(x.get().to_str());
}

fn f(x: &mut Option<int>) {
    g(&*x);
}

pub fn main() {
    let mut x = ~Some(3);
    f(x);
}
