fn main() {
    let a: [u8; 3] = [1,2,3];
    let b = &a;
    let c = b as *const [u32; 3]; //~ ERROR mismatched types [E0308]
}
