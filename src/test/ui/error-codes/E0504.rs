struct FancyNum {
    num: u8,
}

fn main() {
    let fancy_num = FancyNum { num: 5 };
    let fancy_ref = &fancy_num;

    let x = move || {
        println!("child function: {}", fancy_num.num); //~ ERROR E0504
    };

    x();
    println!("main function: {}", fancy_ref.num);
}
