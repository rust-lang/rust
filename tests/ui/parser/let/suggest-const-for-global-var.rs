let X: i32 = 12;
//~^ ERROR `let` statements are not allowed outside of functions or const blocks

fn main() {
    println!("{}", X);
}
