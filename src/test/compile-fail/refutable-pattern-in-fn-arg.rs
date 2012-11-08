fn main() {
    let f = |3: int| io::println("hello");  //~ ERROR refutable pattern
    f(4);
}

