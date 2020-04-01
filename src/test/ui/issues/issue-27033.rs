fn main() {
    match Some(1) {
        None @ _ => {} //~ ERROR match bindings cannot shadow unit variants
    };
    const C: u8 = 1;
    match 1 {
        C @ 2 => { //~ ERROR match bindings cannot shadow constant
            println!("{}", C);
        }
        _ => {}
    };
}
