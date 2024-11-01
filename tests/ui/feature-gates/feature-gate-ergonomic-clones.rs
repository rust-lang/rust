fn ergonomic_clone(x: i32) -> i32 {
    x.use
    //~^ ERROR `.use` calls are experimental [E0658]
}

fn ergonomic_closure_clone() {
    let s1 = String::from("hi!");

    let s2 = use || {
        //~^ ERROR `.use` calls are experimental [E0658]
        s1
    };

    let s3 = use || {
        //~^ ERROR `.use` calls are experimental [E0658]
        //~| ERROR use of moved value: `s1` [E0382]
        s1
    };
}

fn main() {}
