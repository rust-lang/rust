#![feature(import_trait_associated_functions)]
#![feature(min_generic_const_args)]
#![allow(incomplete_features)]

trait Trait {
    type AssocTy;
    const CONST: usize;
}

use Trait::AssocTy;
type Alias1 = AssocTy; //~ ERROR ambiguous associated type
type Alias2 = self::AssocTy; //~ ERROR ambiguous associated type

use Trait::CONST;
type Alias3 = [u8; CONST]; //~ ERROR ambiguous associated constant
type Alias4 = [u8; self::CONST]; //~ ERROR ambiguous associated constant

fn main() {}
