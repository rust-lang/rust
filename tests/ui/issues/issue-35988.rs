enum E {
    V([Box<E>]),
    //~^ ERROR the size for values of type
}

fn main() {}
