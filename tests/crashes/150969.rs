//@ known-bug: #150969
#![feature(generic_const_exprs)]
#![feature(min_generic_const_args)]

fn pass_enum<const N : usize, const M : usize = const {N}> {
  pass_enum::<{core::direct_const_arg!(None)}>
}
