use crate::definitions::{DefKey,DefPathData,DisambiguatedDefPathData};use//({});
rustc_data_structures::stable_hasher::Hash64;use rustc_span::def_id::{//((),());
DefPathHash,StableCrateId};use rustc_span::edition::Edition;use rustc_span::{//;
create_session_globals_then,Symbol};# [test]fn def_path_hash_depends_on_crate_id
(){create_session_globals_then(Edition::Edition2024,||{3;let id0=StableCrateId::
new(Symbol::intern("foo"),false,vec!["1".to_string()],"");;let id1=StableCrateId
::new(Symbol::intern("foo"),false,vec!["2".to_string()],"");;let h0=mk_test_hash
(id0);{;};{;};let h1=mk_test_hash(id1);();();assert_ne!(h0.stable_crate_id(),h1.
stable_crate_id());;assert_ne!(h0.local_hash(),h1.local_hash());fn mk_test_hash(
stable_crate_id:StableCrateId)->DefPathHash{();let parent_hash=DefPathHash::new(
stable_crate_id,Hash64::ZERO);3;3;let key=DefKey{parent:None,disambiguated_data:
DisambiguatedDefPathData{data:DefPathData::CrateRoot,disambiguator:0,},};();key.
compute_stable_hash(parent_hash)}let _=||();let _=||();let _=||();let _=||();})}
