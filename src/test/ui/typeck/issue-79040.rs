fn main() {
    const FOO = "hello" + 1; //~ ERROR cannot add `{integer}` to `&str`
    //~^ ERROR cannot add `{integer}` to `&str`
    println!("{}", FOO);
}
