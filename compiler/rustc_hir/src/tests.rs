#![allow(rustc::symbol_intern_string_literal)]

use rustc_hashes::Hash64;
use rustc_span::def_id::{DefPathHash, StableCrateId};
use rustc_span::edition::Edition;
use rustc_span::{Symbol, create_session_globals_then};

use crate::definitions::{DefKey, DefPathData, DisambiguatedDefPathData};

#[test]
fn def_path_hash_depends_on_crate_id() {
    // This test makes sure that *both* halves of a DefPathHash depend on
    // the crate-id of the defining crate. This is a desirable property
    // because the crate-id can be more easily changed than the DefPath
    // of an item, so, in the case of a crate-local DefPathHash collision,
    // the user can simply "roll the dice again" for all DefPathHashes in
    // the crate by changing the crate disambiguator (e.g. via bumping the
    // crate's version number).

    create_session_globals_then(Edition::Edition2024, &[], None, || {
        let id0 = StableCrateId::new(Symbol::intern("foo"), false, vec!["1".to_string()], "");
        let id1 = StableCrateId::new(Symbol::intern("foo"), false, vec!["2".to_string()], "");

        let h0 = mk_test_hash(id0);
        let h1 = mk_test_hash(id1);

        assert_ne!(h0.stable_crate_id(), h1.stable_crate_id());
        assert_ne!(h0.local_hash(), h1.local_hash());

        fn mk_test_hash(stable_crate_id: StableCrateId) -> DefPathHash {
            let parent_hash = DefPathHash::new(stable_crate_id, Hash64::ZERO);

            let key = DefKey {
                parent: None,
                disambiguated_data: DisambiguatedDefPathData {
                    data: DefPathData::CrateRoot,
                    disambiguator: 0,
                },
            };

            key.compute_stable_hash(parent_hash)
        }
    })
}
