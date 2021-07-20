struct NInts<const N: usize>([u8; N]);
impl NInts<const N: usize> {} //~ ERROR unexpected `const` parameter declaration

fn main() {
    let _: () = 42; //~ ERROR mismatched types
}
