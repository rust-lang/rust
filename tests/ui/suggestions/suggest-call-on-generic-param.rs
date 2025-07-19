fn return_type<T>(t: T) {
    let x: u32 = t(1);
    //~^ ERROR: expected function, found `T` [E0618]
}

fn unknown_return_type<T>(t: T) {
    let x = t();
    //~^ ERROR: expected function, found `T` [E0618]
}

fn nested_return_type<T>(t: Vec<T>) {
    t();
    //~^ ERROR: expected function, found `Vec<T>` [E0618]
}

fn no_return_type<T>(t: T) {
    t(1, 2, true);
    //~^ ERROR: expected function, found `T` [E0618]
}

fn existing_bound<T: Copy>(t: T) {
    t(false);
    //~^ ERROR: expected function, found `T` [E0618]
}

fn main() {}
