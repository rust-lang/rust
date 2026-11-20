//@ run-rustfix
mod test {
    public const X: i32 = 123;
    //~^ ERROR expected one of `!` or `::`, found keyword `const`
}

fn main() {
    println!("{}", test::X);
}
