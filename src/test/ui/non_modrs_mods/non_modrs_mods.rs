// run-pass
//
// ignore-pretty issue #37195
pub mod modrs_mod;
pub mod foors_mod;
#[path = "some_crazy_attr_mod_dir/arbitrary_name.rs"]
pub mod attr_mod;
pub fn main() {
    modrs_mod::inner_modrs_mod::innest::foo();
    modrs_mod::inner_foors_mod::innest::foo();
    modrs_mod::inline::innie::foo();
    foors_mod::inner_modrs_mod::innest::foo();
    foors_mod::inner_foors_mod::innest::foo();
    foors_mod::inline::innie::foo();
    attr_mod::inner_modrs_mod::innest::foo();
}
