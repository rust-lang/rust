//

//@ compile-flags: -g  --remap-path-prefix={{cwd}}=/the/aux-cwd --remap-path-prefix={{test-suite-src-base}}/remap_path_prefix/auxiliary=/the/aux-src

#[inline]
pub fn some_aux_function() -> i32 {
    1234
}
