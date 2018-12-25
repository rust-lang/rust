static S : u64 = { { panic!("foo"); 0 } };
//~^ ERROR panicking in statics is unstable

fn main() {
    println!("{:?}", S);
}
