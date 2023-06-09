static S : u64 = { { panic!("foo"); 0 } };
//~^ ERROR could not evaluate static initializer

fn main() {
    println!("{:?}", S);
}
