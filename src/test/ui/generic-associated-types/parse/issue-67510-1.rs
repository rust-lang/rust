// Several parsing errors for associated type constraints that are supposed
// to trigger all errors in ´get_assoc_type_with_generics´ and
// ´get_generic_args_from_path_segment´ 

#![feature(generic_associated_types)]

trait X {
    type Y<'a>;
}

trait Z {}

impl<T : X<X::Y<'a> = &'a u32>> Z for T {} 
    //~^ ERROR: associated types cannot contain multiple path segments 

fn main() {}
