fn main() {
    const FOO = "hello" + 1;
    //~^ ERROR: can't add `{integer}` to `&str`
    //~| ERROR: missing type for `const` item
    println!("{}", FOO);
}
