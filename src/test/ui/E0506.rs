// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

struct FancyNum {
    num: u8,
}

fn main() {
    let mut fancy_num = FancyNum { num: 5 };
    let fancy_ref = &fancy_num;
    fancy_num = FancyNum { num: 6 }; //[ast]~ ERROR E0506
                                     //[mir]~^ ERROR [E0506]

    println!("Num: {}, Ref: {}", fancy_num.num, fancy_ref.num);
}
