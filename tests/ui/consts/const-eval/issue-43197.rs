const fn foo(x: u32) -> u32 {
    x
}

fn main() {
    const X: u32 = 0 - 1;
    //~^ ERROR constant
    const Y: u32 = foo(0 - 1);
    //~^ ERROR constant
    println!("{} {}", X, Y);
}
