#[allow(dead_code)]
fn main() {
    auto n = 0;//~ ERROR invalid variable declaration
    //~^ HELP to introduce a variable, write `let` instead of `auto`
    auto m;//~ ERROR invalid variable declaration
    //~^ HELP to introduce a variable, write `let` instead of `auto`
    m = 0;
}
