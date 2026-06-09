fn add_one(x: i32) -> i32 {
    x + 1
}

fn add_two(x: i32, y: i32) -> i32 {
    x + y
}

fn main() {
    add_one(2, 2); //~ ERROR this function takes 1 argument but 2 arguments were supplied
    add_one(no_such_local, 10); //~ ERROR cannot find value `no_such_local` in this scope
    //~| ERROR this function takes 1 argument but 2 arguments were supplied
    add_one(10, no_such_local); //~ ERROR cannot find value `no_such_local` in this scope
    //~| ERROR this function takes 1 argument but 2 arguments were supplied
    add_two(10, no_such_local, 10); //~ ERROR cannot find value `no_such_local` in this scope
    //~| ERROR this function takes 2 arguments but 3 arguments were supplied
    add_two(no_such_local, 10, 10); //~ ERROR cannot find value `no_such_local` in this scope
    //~| ERROR this function takes 2 arguments but 3 arguments were supplied
    add_two(10, 10, no_such_local); //~ ERROR cannot find value `no_such_local` in this scope
    //~| ERROR this function takes 2 arguments but 3 arguments were supplied
}
