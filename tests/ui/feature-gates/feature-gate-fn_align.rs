#![crate_type = "lib"]

#[repr(align(16))] //~ ERROR `repr(align)` attributes on functions are unstable
fn requires_alignment() {}

trait MyTrait {
    #[repr(align)] //~ ERROR invalid `repr(align)` attribute: `align` needs an argument
    fn myfun();
}
