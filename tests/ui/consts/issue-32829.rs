static S : u64 = { { panic!("foo"); 0 } };
//~^ ERROR evaluation panicked: foo

fn main() {
    println!("{:?}", S);
}
