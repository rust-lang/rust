trait Tr {
    const C: Self;
}

fn main() {
    let a: u8 = Tr::C; //~ ERROR the trait bound `u8: Tr` is not satisfied
}
