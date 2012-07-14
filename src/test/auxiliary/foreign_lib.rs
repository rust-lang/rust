#[link(name="foreign_lib", vers="0.0")];

extern mod rustrt {
    fn last_os_error() -> ~str;
}