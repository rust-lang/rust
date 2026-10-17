//@ run-pass
//@ run-rustfix

#![deny(rust_2021_incompatible_closure_captures)]
#![allow(unused)]

// Test cases for types that implement an insignificant drop (stlib defined)

macro_rules! test_insig_dtor_for_type {
    ($t: ty, $disambiguator: ident) => {
        mod $disambiguator {
            use std::collections::*;
            use std::rc::Rc;
            use std::sync::Mutex;

            fn test_for_type(t: $t) {
                let tup = (Mutex::new(0), t);

                let _c = || tup.0;
            }
        }
    };
}

test_insig_dtor_for_type!(i32, prim_i32);
test_insig_dtor_for_type!(Vec<i32>, vec_i32);
test_insig_dtor_for_type!(String, string);
test_insig_dtor_for_type!(Vec<String>, vec_string);
test_insig_dtor_for_type!(HashMap<String, String>, hash_map);
test_insig_dtor_for_type!(BTreeMap<String, i32>, btree_map);
test_insig_dtor_for_type!(LinkedList<String>, linked_list);
test_insig_dtor_for_type!(Rc<i32>, rc_i32);
test_insig_dtor_for_type!(Rc<String>, rc_string);
test_insig_dtor_for_type!(std::vec::IntoIter<String>, vec_into_iter);
test_insig_dtor_for_type!(btree_map::IntoIter<String, String>, btree_map_into_iter);
test_insig_dtor_for_type!(std::array::IntoIter<String, 5>, array_into_iter);

fn main() {}
