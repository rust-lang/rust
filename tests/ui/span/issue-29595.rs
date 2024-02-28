trait Tr {
    const C: Self;
}

fn main() {
    let a: u8 = Tr::C; //~ ERROR trait `Tr` is not implemented for `u8`
}
