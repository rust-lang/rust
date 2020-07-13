fn f<'a, 'b>(y: &'b ()) {
    let x: &'a _ = &y;
    //~^ E0490
    //~| E0495
    //~| E0495
}

fn main() {}
