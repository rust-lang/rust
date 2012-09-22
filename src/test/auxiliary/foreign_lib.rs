#[link(name="foreign_lib", vers="0.0")];

extern mod rustrt {
    #[legacy_exports];
    fn last_os_error() -> ~str;
}