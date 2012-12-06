use io::ReaderUtil;
fn main() {
    let mut x: bool = false;
    // this line breaks it
    vec::rusti::move_val_init(&mut x, false);

    let input = io::stdin();
    let line = input.read_line(); // use core's io again

    io::println(fmt!("got %?", line));
}
