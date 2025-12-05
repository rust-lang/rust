trait StreamingIterator {
    type Item<'a>;
    fn next(&mut self) -> Option<Self::Item>;
    //~^ ERROR missing generics for associated type
}

fn main() {}

// call stack from back to front:
// create_substs_for_assoc_ty -> qpath_to_ty -> res_to_ty -> ast_ty_to_ty -> ty_of_fn
