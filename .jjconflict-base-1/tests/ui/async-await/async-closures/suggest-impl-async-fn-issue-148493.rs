#![allow(dead_code)]
//@ run-rustfix
//@ edition: 2021

// The suggestion should be `impl AsyncFn()` instead of something like `{async closure@...}`

fn test1() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    //~| HELP replace with an appropriate return type
    //~| SUGGESTION impl AsyncFn()
    async || {}
}

fn test2() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    //~| HELP replace with an appropriate return type
    //~| SUGGESTION impl AsyncFn(i32) -> i32
    async |x: i32| x + 1
}

fn test3() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    //~| HELP replace with an appropriate return type
    //~| SUGGESTION impl AsyncFn(i32, i32) -> i32
    async |x: i32, y: i32| x + y
}

fn test4() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    //~| HELP replace with an appropriate return type
    //~| SUGGESTION impl AsyncFn()
    async || -> () { () }
}

fn test5() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    //~| HELP replace with an appropriate return type
    //~| SUGGESTION impl AsyncFn() -> i32
    let z = 42;
    async move || z
}

fn test6() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    //~| HELP replace with an appropriate return type
    //~| SUGGESTION impl AsyncFnMut() -> i32
    let mut x = 0;
    async move || {
        x += 1;
        x
    }
}

fn test7() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    //~| HELP replace with an appropriate return type
    //~| SUGGESTION impl AsyncFnOnce()
    let s = String::from("hello");
    async move || {
        drop(s);
    }
}

fn main() {}
