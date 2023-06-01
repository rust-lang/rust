// revisions: current next
//[next] compile-flags: -Ztrait-solver=next
// check-pass

#![feature(type_alias_impl_trait)]

fn main() {
    type T = impl Copy;
    let foo: T = (1u32, 2u32);
    let x: (_, _) = foo;
    println!("{:?}", x);
}
