fn f<'r>(p: &'r mut fn(p: &mut ())) {
    (*p)(()) //~  ERROR mismatched types
             //~| NOTE expected `&mut ()`, found `()`
             //~| NOTE arguments to this function are incorrect
}

fn main() {}
