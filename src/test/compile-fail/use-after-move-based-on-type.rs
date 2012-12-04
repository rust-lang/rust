fn main() {
    let x = ~"Hello!";
    let _y = x;
    io::println(x); //~ ERROR use of moved variable
}

