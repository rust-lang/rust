// Issue #11669

// ignore-tidy-cr

fn main() {
    // \r\n
    let ok = "This is \
 a test";
    // \r only
    let bad = "This is \ a test";
    //~^ ERROR unknown character escape: \r
    //~^^ HELP this is an isolated carriage return

}
