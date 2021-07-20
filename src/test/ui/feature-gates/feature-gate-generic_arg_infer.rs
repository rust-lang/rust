// run-pass
#![feature(generic_arg_infer)]

fn foo<const N: usize>(_: [u8; N]) -> [u8; N] {
  [0; N]
}

fn main() {
    let _x = foo::<_>([1,2]);
}
