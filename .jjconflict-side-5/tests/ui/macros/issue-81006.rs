//@ check-fail

// First format below would cause a panic, second would generate error with incorrect span

fn main() {
    let _ = format!("→{}→\n");
    //~^ ERROR 1 positional argument in format string, but no arguments were given
    let _ = format!("→{} \n");
    //~^ ERROR 1 positional argument in format string, but no arguments were given
}
