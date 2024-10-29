fn ergonomic_clone(x: i32) -> i32 {
    x.use
    //~^ ERROR expected identifier, found keyword `use`
    //~| ERROR `i32` is a primitive type and therefore doesn't have fields [E0610]
}

fn ergonomic_closure_clone() {
    let s1 = String::from("hi!");

    let s2 = use || {
        //~^ ERROR expected expression, found keyword `use`
        s1
    };

    let s3 = use || {
        s1
    };
}

fn main() {}
