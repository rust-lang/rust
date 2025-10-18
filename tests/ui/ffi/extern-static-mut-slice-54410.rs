// https://github.com/rust-lang/rust/issues/54410
extern "C" {
    pub static mut symbol: [i8];
    //~^ ERROR the size for values of type `[i8]` cannot be known at compilation time
}

fn main() {
    println!("{:p}", unsafe { &symbol });
}
