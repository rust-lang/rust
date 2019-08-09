struct FancyNum {
    num: u8,
}

fn main() {
    let mut fancy = FancyNum{ num: 5 };
    let fancy_ref = &(&mut fancy);
    fancy_ref.num = 6; //~ ERROR cannot assign to `fancy_ref.num` which is behind a `&` reference
    println!("{}", fancy_ref.num);
}
