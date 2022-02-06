struct X<const N: usize = {
    let s: &'static str; s.len()
    //~^ ERROR borrow of possibly-uninitialized variable
}>;

fn main() {}
