fn return_type<T>(t: T) {
    let x: u32 = t(1);
    //~^ ERROR: expected function, found `T` [E0618]
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
