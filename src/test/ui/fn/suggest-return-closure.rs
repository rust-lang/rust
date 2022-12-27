fn fn_once() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types [E0121]
    //~| NOTE not allowed in type signatures
    //~| HELP replace with an appropriate return type
    //~| SUGGESTION impl FnOnce()
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
    let x = String::new();
    |c| {
        x.push(c);
    }
}

fn fun() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types [E0121]
    //~| NOTE not allowed in type signatures
    //~| HELP replace with an appropriate return type
    //~| SUGGESTION impl Fn() -> i32
    || 1i32
}

fn main() {}
