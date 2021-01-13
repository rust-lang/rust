#![allow(unused_assignments, unused_variables)]

extern crate used_crate;

fn main() {
    used_crate::used_function();
    let some_vec = vec![1, 2, 3, 4];
    used_crate::used_only_from_bin_crate_generic_function(&some_vec);
    used_crate::used_only_from_bin_crate_generic_function("used from bin uses_crate.rs");
    used_crate::used_from_bin_crate_and_lib_crate_generic_function(some_vec);
    used_crate::used_with_same_type_from_bin_crate_and_lib_crate_generic_function("interesting?");
}
