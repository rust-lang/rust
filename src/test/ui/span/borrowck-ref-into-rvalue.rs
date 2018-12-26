fn main() {
    let msg;
    match Some("Hello".to_string()) {
        Some(ref m) => {
        //~^ ERROR borrowed value does not live long enough
            msg = m;
        },
        None => { panic!() }
    }
    println!("{}", *msg);
}
