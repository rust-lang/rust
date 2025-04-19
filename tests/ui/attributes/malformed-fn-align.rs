#![crate_type = "lib"]

trait MyTrait {
    #[repr(align)] //~ ERROR invalid `repr(align)` attribute: `align` needs an argument
    fn myfun1();

    #[repr(align(1, 2))] //~ ERROR incorrect `repr(align)` attribute format: `align` takes exactly one argument in parentheses
    fn myfun2();
}
