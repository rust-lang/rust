fn main() {
    const FOO: i32 = 123;
    println!("{foo:X}");
    //~^ ERROR: cannot find value `foo` in this scope
    println!("{:.foo$}", 0);
    //~^ ERROR: cannot find value `foo` in this scope
}
