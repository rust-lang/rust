#![allow(unused_assignments, unused_variables)]

mod used_crate;

fn main() {
    used_crate::used_function();
    let some_vec = vec![1, 2, 3, 4];
    used_crate::used_generic_function(&some_vec);
    used_crate::used_twice_generic_function(some_vec);
}
