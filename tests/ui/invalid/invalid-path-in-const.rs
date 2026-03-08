fn main() {
    fn f(a: [u8; u32::DOESNOTEXIST]) {}
    //~^ ERROR no associated function or constant named `DOESNOTEXIST` found for type `u32`
}
