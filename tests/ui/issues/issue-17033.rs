fn f<'r>(p: &'r mut fn(p: &mut ())) {
    (*p)(()) //~  ERROR mismatched types
             //~| NOTE_NONVIRAL expected `&mut ()`, found `()`
}

fn main() {}
