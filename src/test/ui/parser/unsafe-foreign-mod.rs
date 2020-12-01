unsafe extern {
    //~^ ERROR extern block cannot be declared unsafe
}

unsafe extern "C" {
    //~^ ERROR extern block cannot be declared unsafe
}

fn main() {}
