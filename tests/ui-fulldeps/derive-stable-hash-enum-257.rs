#![feature(rustc_private)]
//@ edition: 2024
//@ run-pass

// Test that `#[derive(StableHash)]` supports enums with more than 256 variants.

use std::collections::HashMap;

use rustc_data_structures::stable_hash::{DummyStableHashCtxt, StableHash, StableHasher};
use rustc_hashes::Hash128;
use rustc_macros::StableHash;

extern crate rustc_data_structures;
#[expect(
    unused_extern_crates,
    reason = "Necessary to pull in object code, as other rustc crates are only shipped as rmeta"
)]
extern crate rustc_driver;
extern crate rustc_hashes;
extern crate rustc_macros;

macro_rules! enum_with_all_variants {
    (
        $( #[$meta:meta] )*
        enum $Enum:ident {
            $( $Variant:ident ),* $(,)?
        }
    ) => {
        $( #[$meta] )*
        enum $Enum {
            $( $Variant, )*
        }

        impl $Enum {
            const ALL_VARIANTS: &[Self] = &[
                $( Self::$Variant, )*
            ];
        }
    }
}

enum_with_all_variants! {
    #[derive(Clone, Copy, Debug, StableHash)]
    enum Enum257 {
        V000, V001, V002, V003, V004, V005, V006, V007, V008, V009,
        V010, V011, V012, V013, V014, V015, V016, V017, V018, V019,
        V020, V021, V022, V023, V024, V025, V026, V027, V028, V029,
        V030, V031, V032, V033, V034, V035, V036, V037, V038, V039,
        V040, V041, V042, V043, V044, V045, V046, V047, V048, V049,
        V050, V051, V052, V053, V054, V055, V056, V057, V058, V059,
        V060, V061, V062, V063, V064, V065, V066, V067, V068, V069,
        V070, V071, V072, V073, V074, V075, V076, V077, V078, V079,
        V080, V081, V082, V083, V084, V085, V086, V087, V088, V089,
        V090, V091, V092, V093, V094, V095, V096, V097, V098, V099,

        V100, V101, V102, V103, V104, V105, V106, V107, V108, V109,
        V110, V111, V112, V113, V114, V115, V116, V117, V118, V119,
        V120, V121, V122, V123, V124, V125, V126, V127, V128, V129,
        V130, V131, V132, V133, V134, V135, V136, V137, V138, V139,
        V140, V141, V142, V143, V144, V145, V146, V147, V148, V149,
        V150, V151, V152, V153, V154, V155, V156, V157, V158, V159,
        V160, V161, V162, V163, V164, V165, V166, V167, V168, V169,
        V170, V171, V172, V173, V174, V175, V176, V177, V178, V179,
        V180, V181, V182, V183, V184, V185, V186, V187, V188, V189,
        V190, V191, V192, V193, V194, V195, V196, V197, V198, V199,

        V200, V201, V202, V203, V204, V205, V206, V207, V208, V209,
        V210, V211, V212, V213, V214, V215, V216, V217, V218, V219,
        V220, V221, V222, V223, V224, V225, V226, V227, V228, V229,
        V230, V231, V232, V233, V234, V235, V236, V237, V238, V239,
        V240, V241, V242, V243, V244, V245, V246, V247, V248, V249,
        V250, V251, V252, V253, V254, V255, V256,
    }
}

fn main() {
    assert_eq!(Enum257::ALL_VARIANTS.len(), 257);

    let mut hashes = HashMap::<Hash128, Enum257>::with_capacity(257);

    for &variant in Enum257::ALL_VARIANTS {
        let hcx = &mut DummyStableHashCtxt;
        let mut hasher = StableHasher::new();
        variant.stable_hash(hcx, &mut hasher);
        let hash = hasher.finish();

        if let Some(old) = hashes.insert(hash, variant) {
            panic!("StableHash collision between {old:?} and {variant:?}: {hash:?}");
        }
    }

    assert_eq!(hashes.len(), 257);
}
