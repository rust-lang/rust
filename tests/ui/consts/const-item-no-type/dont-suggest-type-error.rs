fn main() {
    const FOO = "hello" + 1;
    //~^ ERROR cannot add `{integer}` to `&str`
    //~| ERROR missing type for `const` item
    println!("{}", FOO);
}
