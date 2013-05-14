fn main() {
    let x = ~"Hello world!";
    do task::spawn {
        io::println(x);
    }
    io::println(x); //~ ERROR use of moved value
}
