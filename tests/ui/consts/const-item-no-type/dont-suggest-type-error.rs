fn main() {
    const FOO: _ = "hello" + 1;
    //~^ ERROR cannot add `{integer}` to `&str`
    //~| ERROR the placeholder `_` is not allowed within types on item signatures for constants [E0121]
    println!("{}", FOO);
}
