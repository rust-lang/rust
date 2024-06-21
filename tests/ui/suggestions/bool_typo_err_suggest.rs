// Suggest the boolean value instead of emit a generic error that the value
// True is not in the scope.

fn main() {
    let x = True;
    //~^ ERROR cannot find value `True`
    //~| HELP you may want to use a bool value instead

    let y = False;
    //~^ ERROR cannot find value `False`
    //~| HELP you may want to use a bool value instead
}
