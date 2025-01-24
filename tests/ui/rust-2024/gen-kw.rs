//@ revisions: e2015 e2018
//@[e2018] edition: 2018

#![deny(rust_2024_compatibility)]

fn gen() {}
//~^ ERROR `gen` is a keyword in the 2024 edition
//[e2015]~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2024!
//[e2018]~| WARNING this is accepted in the current edition (Rust 2018) but is a hard error in Rust 2024!

fn main() {
    let gen = r#gen;
    //~^ ERROR `gen` is a keyword in the 2024 edition
    //[e2015]~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2024!
    //[e2018]~| WARNING this is accepted in the current edition (Rust 2018) but is a hard error in Rust 2024!
}

macro_rules! t {
    () => { mod test { fn gen() {} } }
    //~^ ERROR `gen` is a keyword in the 2024 edition
    //[e2015]~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2024!
    //[e2018]~| WARNING this is accepted in the current edition (Rust 2018) but is a hard error in Rust 2024!
}

fn test<'gen>(_: &'gen i32) {}
//~^ ERROR `gen` is a keyword in the 2024 edition
//~| ERROR `gen` is a keyword in the 2024 edition
//[e2015]~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2024!
//[e2015]~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2024!
//[e2018]~| WARNING this is accepted in the current edition (Rust 2018) but is a hard error in Rust 2024!
//[e2018]~| WARNING this is accepted in the current edition (Rust 2018) but is a hard error in Rust 2024!

struct Test<'gen>(Box<Test<'gen>>, &'gen ());
//~^ ERROR `gen` is a keyword in the 2024 edition
//~| ERROR `gen` is a keyword in the 2024 edition
//~| ERROR `gen` is a keyword in the 2024 edition
//[e2015]~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2024!
//[e2015]~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2024!
//[e2015]~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2024!
//[e2018]~| WARNING this is accepted in the current edition (Rust 2018) but is a hard error in Rust 2024!
//[e2018]~| WARNING this is accepted in the current edition (Rust 2018) but is a hard error in Rust 2024!
//[e2018]~| WARNING this is accepted in the current edition (Rust 2018) but is a hard error in Rust 2024!

t!();
