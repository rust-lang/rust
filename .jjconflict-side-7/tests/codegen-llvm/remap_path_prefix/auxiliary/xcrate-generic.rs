//
//@ compile-flags: -g  --remap-path-prefix={{cwd}}=/the/aux-cwd --remap-path-prefix={{src-base}}/remap_path_prefix/auxiliary=/the/aux-src

#![crate_type = "lib"]

pub fn foo<T: Default>() -> T {
    T::default()
}
