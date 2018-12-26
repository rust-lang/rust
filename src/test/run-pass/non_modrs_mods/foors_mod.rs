//
// ignore-test: not a test, used by non_modrs_mods.rs

pub mod inner_modrs_mod;
pub mod inner_foors_mod;
pub mod inline {
    #[path="somename.rs"]
    pub mod innie;
}
