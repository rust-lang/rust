trait Cat {
    fn nya() {}
}

fn uwu<T: Cat>(c: T) {
    c.nya();
    //~^ ERROR no method named `nya` found for type parameter `T` in the current scope
    //~| SUGGESTION T::nya()
}

fn main() {}
