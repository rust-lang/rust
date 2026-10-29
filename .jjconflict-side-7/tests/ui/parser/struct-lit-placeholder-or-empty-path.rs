fn main() {
    let _ = {foo: (), bar: {} }; //~ ERROR struct literal body without path
    //~| NOTE struct name missing for struct literal
    //~| HELP add the correct type
    let _ = _ {foo: (), bar: {} }; //~ ERROR placeholder `_` is not allowed for the path in struct literals
    //~| NOTE not allowed in struct literals
    //~| HELP replace it with the correct type
    let _ = {foo: ()}; //~ ERROR struct literal body without path
    //~| NOTE struct name missing for struct literal
    //~| HELP add the correct type
    let _ = _ {foo: ()}; //~ ERROR placeholder `_` is not allowed for the path in struct literals
    //~| NOTE not allowed in struct literals
    //~| HELP replace it with the correct type
}
