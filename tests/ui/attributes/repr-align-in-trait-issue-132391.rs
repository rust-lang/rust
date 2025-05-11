trait MyTrait {
    #[repr(align)] //~ ERROR invalid `repr(align)` attribute: `align` needs an argument
    fn myfun();
}

pub fn main() {}
