#[link(name="foreign_lib", vers="0.0")];

native mod rustrt {
    fn last_os_error() -> str;
}