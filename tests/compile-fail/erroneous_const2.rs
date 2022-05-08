const X: u32 = 5;
const Y: u32 = 6;
const FOO: u32 = [X - Y, Y - X][(X < Y) as usize];
//~^ERROR any use of this value
//~|WARN previously accepted

fn main() {
    println!("{}", FOO); //~ERROR post-monomorphization error
    //~|ERROR evaluation of constant value failed
    //~|ERROR erroneous constant used
    //~|WARN previously accepted
}
