fn f() {
    let a = ~"hello";
    let b: &str = a;
    io::println(b);
}

fn g() {
    let c = ~"world";
    let d: &str;
    d = c;
    io::println(d);
}

fn main() {
    f();
    g();
}

