fn main() {
    let a: usize = 0;
    let long_name: usize = 0;

    println!("{}", a as usize > long_name);
    println!("{}", a as usize < long_name); //~ ERROR `<` is interpreted as a start of generic
    println!("{}{}", a as usize < long_name, long_name);
    //~^ ERROR `<` is interpreted as a start of generic
    println!("{}", a as usize < 4); //~ ERROR `<` is interpreted as a start of generic

    println!("{}", a
                   as
                   usize
                   < //~ ERROR `<` is interpreted as a start of generic
                   4);
    println!("{}", a


                   as


                   usize
                   < //~ ERROR `<` is interpreted as a start of generic
                   5);

    println!("{}", a as usize << long_name); //~ ERROR `<<` is interpreted as a start of generic
}
