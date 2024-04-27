//@ run-pass

pub mod inner_modrs_mod;
pub mod inner_foors_mod;
pub mod inline {
    #[path="somename.rs"]
    pub mod innie;
}
