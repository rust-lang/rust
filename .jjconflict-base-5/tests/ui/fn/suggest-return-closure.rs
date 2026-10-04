fn fn_once() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types [E0121]
    //~| NOTE not allowed in type signatures
    //~| HELP replace with an appropriate return type
    //~| SUGGESTION impl FnOnce()
    //~| NOTE for more information on `Fn` traits and closure types
    let x = String::new();
    || {
        drop(x);
    }
}

fn fn_mut() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types [E0121]
    //~| NOTE not allowed in type signatures
    //~| HELP replace with an appropriate return type
    //~| SUGGESTION impl FnMut(char)
    //~| NOTE for more information on `Fn` traits and closure types
    let x = String::new();
    //~^ HELP: consider changing this to be mutable
    //~| NOTE binding `x` declared here
    //~| SUGGESTION mut
    |c| { //~ NOTE: value captured here
        x.push(c);
        //~^ ERROR: does not live long enough
        //~| NOTE: does not live long enough
        //~| NOTE: cannot borrow as mutable
        //~| ERROR: not declared as mutable
    }
} //~ NOTE: borrow later used here
//~^ NOTE: dropped here

fn fun() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types [E0121]
    //~| NOTE not allowed in type signatures
    //~| HELP replace with an appropriate return type
    //~| SUGGESTION impl Fn() -> i32
    //~| NOTE for more information on `Fn` traits and closure types
    || 1i32
}

fn main() {}
