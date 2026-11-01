fn main() {
    hey_i_get_compiled();
}

#[inline(never)]
fn hey_i_get_compiled() {
    println!("Hi! Do or do not strip me, your choice.");
}
