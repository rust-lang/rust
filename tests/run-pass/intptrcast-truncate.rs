// compile-flags: -Zmiri-seed=0000000000000000

fn main() {
    let x = &42 as *const i32 as usize; 
    let y = x * 2;
    let z = y as u8 as usize;
    assert_eq!(z, y % 256);
}
