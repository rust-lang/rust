fn f<'r>(p: &'r mut fn(p: &mut ())) {
    (*p)(()) //~  ERROR mismatched types
             //~| expected `&mut ()`, found `()`
}

fn main() {}
