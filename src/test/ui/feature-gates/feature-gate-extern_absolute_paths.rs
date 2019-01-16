use core::default;

fn main() {
    let _: u8 = ::core::default::Default();
    //~^ ERROR expected function, found trait `core::default::Default`
}
