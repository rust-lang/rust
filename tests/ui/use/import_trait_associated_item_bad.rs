#![feature(import_trait_associated_functions)]
#![feature(min_generic_const_args)]
#![allow(incomplete_features)]

trait Trait {
    type AssocTy;
    const CONST: usize;
}

use Trait::AssocTy;
type Alias1 = AssocTy; //~ ERROR cannot infer type, type annotations needed

use Trait::CONST;
type Alias2 = [u8; CONST]; //~ ERROR cannot infer type, type annotations needed

fn main() {}
