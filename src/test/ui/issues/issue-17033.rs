fn f<'r>(p: &'r mut fn(p: &mut ())) {
    (*p)(()) //~  ERROR mismatched types
             //~| expected type `&mut ()`
             //~| found type `()`
             //~| expected &mut (), found ()
}

fn main() {}
