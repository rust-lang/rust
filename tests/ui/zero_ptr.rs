#[allow(unused_variables)]
fn main() {
    let x = 0 as *const usize;
    let y = 0 as *mut f64;

    let z = 0;
    let z = z as *const usize; // this is currently not caught
}
