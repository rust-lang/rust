pub fn main() {
    let x = ~"Hello world!";
    do task::spawn {
        io::println(x);
    }
}
