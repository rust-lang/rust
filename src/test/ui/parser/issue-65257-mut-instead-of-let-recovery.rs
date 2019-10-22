
#[allow(dead_code)]
fn main() {
    mut n = 0;//~ ERROR invalid variable declaration
    //~^ HELP missing `let`
    mut var;//~ ERROR invalid variable declaration
    //~^ HELP missing `let`
    var = 0;
}
