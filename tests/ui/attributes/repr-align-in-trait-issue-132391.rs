trait MyTrait {
    #[repr(align)] //~ ERROR malformed `repr` attribute input
    fn myfun();
}

pub fn main() {}
