fn main() {
    let elem = 6i8;
    let e2 = 230;
    //~^ ERROR literal out of range for `i8`
    //~| HELP consider using the type `u8` instead

    let mut vec = Vec::new();

    vec.push(e2);
    vec.push(elem);

    println!("{:?}", vec);
}
