#[cfg_attr(feature = "foo", path = "foo.rs")]
#[cfg_attr(not(feture = "foo"), path = "bar.rs")]
mod sub_mod;

#[cfg_attr(target_arch = "wasm32", path = "dir/dir1/dir2/wasm32.rs")]
#[cfg_attr(not(target_arch = "wasm32"), path = "dir/dir1/dir3/wasm32.rs")]
mod wasm32;

#[some_attr(path = "somewhere.rs")]
mod other;
