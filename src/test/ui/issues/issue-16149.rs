extern {
    static externalValue: isize;
}

fn main() {
    let boolValue = match 42 {
        externalValue => true,
        //~^ ERROR match bindings cannot shadow statics
        _ => false
    };
}
