// run-pass
// ignore-pretty issue #37195

mod mod_dir_implicit_aux;

pub fn main() {
    assert_eq!(mod_dir_implicit_aux::foo(), 10);
}
