// run-pass
// ignore-pretty issue #37195

// Testing that the parser for each file tracks its modules
// and paths independently. The load_another_mod module should
// not try to reuse the 'mod_dir_simple' path.

mod mod_dir_simple {
    pub mod load_another_mod;
}

pub fn main() {
    assert_eq!(mod_dir_simple::load_another_mod::test::foo(), 10);
}
