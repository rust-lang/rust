const fn foo(x: u32) -> u32 {
    x
}

fn main() {
    const X: u32 = 0 - 1;
    //~^ ERROR overflow
    const Y: u32 = foo(0 - 1);
    //~^ ERROR overflow
    println!("{} {}", X, Y);
}
