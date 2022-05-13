#![feature(ptr_metadata)]

use std::ptr::TypedMetadata;

struct Struct;
trait Trait {}

fn main() {
    let struct_metadata: TypedMetadata<Struct> = TypedMetadata(());

    let _: TypedMetadata<dyn Trait> = struct_metadata; //~ ERROR `Struct: Trait` is not satisfied
    let _: TypedMetadata<[Struct]> = struct_metadata; //~ ERROR mismatched types

    let array_metadata: TypedMetadata<[u8; 4]> = TypedMetadata(());

    let _: TypedMetadata<[u32]> = array_metadata; //~ ERROR mismatched types
    let _: TypedMetadata<Struct> = array_metadata; //~ ERROR mismatched types
}
