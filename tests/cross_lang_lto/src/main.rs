/*
 * Compile the C code with:
 * gcc -c -flto add.c -ffat-lto-objects
 * ar rcs libadd.a add.o
 *
 * Compile the Rust code with:
 * CG_RUSTFLAGS="-L native=. -Clinker-plugin-lto -Clinker=gcc" y cargo run --release
 */

#[link(name="add")]
unsafe extern "C" {
    fn my_add(a: u32, b: u32) -> u32;
}

fn main() {
    let res = unsafe { my_add(30, 12) };
    println!("{}", res);
}
