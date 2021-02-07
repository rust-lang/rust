fn f<'r>(p: &'r mut fn(p: &mut ())) {
    (*p)(()) //~  ERROR arguments to this function are incorrect
             //~| expected `&mut ()`, found `()`
}

fn main() {}
