// run-rustfix
fn main () {
    #[allow(non_upper_case_globals)]
    let foo: usize = 42;
    let _: [u8; foo]; //~ ERROR E0435
}
