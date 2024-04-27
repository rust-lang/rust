extern "Rust" {
    fn dstfn(src: i32, dst: err);
    //~^ ERROR cannot find type `err` in this scope
}

fn main() {
    dstfn(1);
    //~^ ERROR function takes 2 arguments but 1 argument was supplied
}
