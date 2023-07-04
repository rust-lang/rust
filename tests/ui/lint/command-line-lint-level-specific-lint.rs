// compile-flags: -A warnings -W unused-variables

fn main() {
    let x = 0u8;
    //~^ WARNING unused variable

    // Source code lint level should override command-line lint level in any case.
    #[deny(unused_variables)]
    let y = 0u8;
    //~^ ERROR unused variable
}
