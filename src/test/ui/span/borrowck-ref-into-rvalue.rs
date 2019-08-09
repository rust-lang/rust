fn main() {
    let msg;
    match Some("Hello".to_string()) {
        //~^ ERROR temporary value dropped while borrowed
        Some(ref m) => {
            msg = m;
        },
        None => { panic!() }
    }
    println!("{}", *msg);
}
