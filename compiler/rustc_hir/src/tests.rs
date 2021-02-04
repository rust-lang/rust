use crate::definitions::{DefKey, DefPathData, DisambiguatedDefPathData};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_span::crate_disambiguator::CrateDisambiguator;
use rustc_span::def_id::{DefPathHash, StableCrateId};

#[test]
fn def_path_hash_depends_on_crate_id() {
    // This test makes sure that *both* halves of a DefPathHash depend on
    // the crate-id of the defining crate. This is a desirable property
    // because the crate-id can be more easily changed than the DefPath
    // of an item, so, in the case of a crate-local DefPathHash collision,
    // the user can simply "role the dice again" for all DefPathHashes in
    // the crate by changing the crate disambiguator (e.g. via bumping the
    // crate's version number).

    let d0 = CrateDisambiguator::from(Fingerprint::new(12, 34));
    let d1 = CrateDisambiguator::from(Fingerprint::new(56, 78));

    let h0 = mk_test_hash("foo", d0);
    let h1 = mk_test_hash("foo", d1);

    assert_ne!(h0.stable_crate_id(), h1.stable_crate_id());
    assert_ne!(h0.local_hash(), h1.local_hash());

    fn mk_test_hash(crate_name: &str, crate_disambiguator: CrateDisambiguator) -> DefPathHash {
        let stable_crate_id = StableCrateId::new(crate_name, crate_disambiguator);
        let parent_hash = DefPathHash::new(stable_crate_id, 0);

        let key = DefKey {
            parent: None,
            disambiguated_data: DisambiguatedDefPathData {
                data: DefPathData::CrateRoot,
                disambiguator: 0,
            },
        };

        key.compute_stable_hash(parent_hash)
    }
}
