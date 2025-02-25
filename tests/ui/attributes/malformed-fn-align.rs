#![feature(fn_align)]
#![crate_type = "lib"]

trait MyTrait {
    #[repr(align)] //~ ERROR invalid `repr(align)` attribute: `align` needs an argument
    fn myfun();
}
