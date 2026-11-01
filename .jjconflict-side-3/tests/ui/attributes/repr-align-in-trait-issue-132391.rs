trait MyTrait {
    #[repr(align)]
    //~^ ERROR malformed `repr` attribute input
    //~| ERROR attribute cannot be used on
    fn myfun();
}

pub fn main() {}
