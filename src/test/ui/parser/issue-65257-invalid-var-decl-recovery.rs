fn main() {
    auto n = 0;//~ ERROR invalid variable declaration
    //~^ HELP to introduce a variable, write `let` instead of `auto`
    auto m;//~ ERROR invalid variable declaration
    //~^ HELP to introduce a variable, write `let` instead of `auto`
    m = 0;

    var n = 0;//~ ERROR invalid variable declaration
    //~^ HELP to introduce a variable, write `let` instead of `var`
    var m;//~ ERROR invalid variable declaration
    //~^ HELP to introduce a variable, write `let` instead of `var`
    m = 0;

    mut n = 0;//~ ERROR invalid variable declaration
    //~^ HELP missing `let`
    mut var;//~ ERROR invalid variable declaration
    //~^ HELP missing `let`
    var = 0;

    let _recovery_witness: () = 0; //~ ERROR mismatched types
}
