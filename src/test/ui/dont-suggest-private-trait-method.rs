struct T;

fn main() {
    T::new();
    //~^ ERROR no function or associated item named `new` found for type `T` in the current scope
}
