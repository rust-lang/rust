// compile-flags: -Zmiri-seed=0000000000000000

fn main() {
    // Some casting-to-int with arithmetic.
    let x = &42 as *const i32 as usize; 
    let y = x * 2;
    assert_eq!(y, x + x);
    let z = y as u8 as usize;
    assert_eq!(z, y % 256);

    // Pointer string formatting! We can't check the output as it changes when libstd changes,
    // but we can make sure Miri does not error.
    format!("{:?}", &mut 13 as *mut _);
}
