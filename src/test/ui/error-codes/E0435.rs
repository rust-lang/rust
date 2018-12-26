fn main () {
    let foo = 42u32;
    let _: [u8; foo]; //~ ERROR E0435
}
