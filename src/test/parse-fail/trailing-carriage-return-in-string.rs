// compile-flags: -Z parse-only

// ignore-tidy-cr
// Issue #11669

fn main() {
    // \r\n
    let ok = "This is \
 a test";
    // \r only
    let bad = "This is \ a test";
    //~^ ERROR unknown character escape: \r
    //~^^ HELP this is an isolated carriage return

}
