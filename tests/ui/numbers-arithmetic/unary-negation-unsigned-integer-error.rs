fn main() {
    let x = -1 as usize; //~ ERROR: cannot apply unary operator `-`
    let x = (-1) as usize; //~ ERROR: cannot apply unary operator `-`
    let x: u32 = -1; //~ ERROR: cannot apply unary operator `-`
}
