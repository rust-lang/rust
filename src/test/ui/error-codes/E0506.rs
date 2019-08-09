struct FancyNum {
    num: u8,
}

fn main() {
    let mut fancy_num = FancyNum { num: 5 };
    let fancy_ref = &fancy_num;
    fancy_num = FancyNum { num: 6 }; //~ ERROR [E0506]

    println!("Num: {}, Ref: {}", fancy_num.num, fancy_ref.num);
}
