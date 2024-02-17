//@ compile-flags: -Zdeduplicate-diagnostics=yes
//@ run-rustfix

fn main() {
    #[cfg(key=foo)]
    //~^ ERROR expected unsuffixed literal, found `foo`
    //~| HELP surround the identifier with quotation marks to parse it as a string
    println!();
    #[cfg(key="bar")]
    println!();
    #[cfg(key=foo bar baz)]
    //~^ ERROR expected unsuffixed literal, found `foo`
    //~| HELP surround the identifier with quotation marks to parse it as a string
    println!();
}
