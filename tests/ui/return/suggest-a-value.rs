fn tuple_returned() -> (i32,) {
    return;
    //~^ ERROR `return;` in a function whose return type is not `()`
}

fn string_returned() -> String {
    return;
    //~^ ERROR `return;` in a function whose return type is not `()`
}

fn vec_returned() -> Vec<u8> {
    return;
    //~^ ERROR `return;` in a function whose return type is not `()`
}

fn main() {}
