// Compiler:
//
// Run-time:
//   status: 0

fn main() {
    let mut value = (1, 1);
    let ptr = &mut value as *mut (i32, i32);
    println!("{:?}", ptr.wrapping_offset(10));
}
