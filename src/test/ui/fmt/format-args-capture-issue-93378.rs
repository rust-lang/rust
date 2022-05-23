fn main() {
    let a = "a";
    let b = "b";

    println!("{a} {b} {} {} {c} {}", c = "c");
    //~^ ERROR: invalid reference to positional arguments 1 and 2 (there is 1 argument)

    let n = 1;
    println!("{a:.n$} {b:.*}");
    //~^ ERROR: invalid reference to positional argument 0 (no arguments were given)
}
