
#[allow(dead_code)]
fn main() {
    var n = 0;//~ ERROR invalid variable declaration
    //~^ HELP to introduce a variable, write `let` instead of `var`
    var m;//~ ERROR invalid variable declaration
    //~^ HELP to introduce a variable, write `let` instead of `var`
    m = 0;
}
