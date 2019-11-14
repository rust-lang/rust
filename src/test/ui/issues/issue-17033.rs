fn f<'r>(p: &'r mut fn(p: &mut ())) {
    (*p)(()) //~  ERROR mismatched types
             //~| expected mutable reference `&mut ()`
             //~| found unit type `()`
             //~| expected &mut (), found ()
}

fn main() {}
