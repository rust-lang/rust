// build-pass
#![feature(ptr_metadata)]

use std::ptr::TypedMetadata;

struct Struct;
trait Trait {}

impl Trait for Struct {}

fn main() {
    // array -> slice
    let sized: TypedMetadata<[u8; 5]> = TypedMetadata(());
    let _: TypedMetadata<[u8]> = sized;

    // sized -> dyn
    let sized: TypedMetadata<Struct> = TypedMetadata(());
    let dyn_trait: TypedMetadata<dyn Trait + Sync> = sized;

    // dyn -> dyn
    let _: TypedMetadata<dyn Trait> = dyn_trait;

    // identity
    let sized: TypedMetadata<Struct> = TypedMetadata(());
    let _ = sized as TypedMetadata<Struct>;
}
