fn main() {
    let x = 3;
    debug!("&x=%x", ptr::to_uint(&x));
}