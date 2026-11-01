macro_rules! dot {
    ($id:ident) => {
        ().$id();
    }
}

macro_rules! dispatch {
    ($id:ident) => {
        <()>::$id();
    }
}

fn main() {
    dot!(hello);
    //~^ ERROR no method named `hello` found for unit type `()` in the current scope
    dispatch!(hello);
    //~^ ERROR no associated function or constant named `hello` found for unit type `()` in the current scope
}
