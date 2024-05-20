//@ compile-flags: -O
// XXX: The x86-64 assembly get optimized correclty. But llvm-ir output is not until llvm 18?
//@ min-llvm-version: 18

#![crate_type = "lib"]

pub enum K{
    A(Box<[i32]>),
    B(Box<[u8]>),
    C(Box<[String]>),
    D(Box<[u16]>),
}

#[no_mangle]
// CHECK-LABEL: @get_len
// CHECK: getelementptr inbounds
// CHECK-NEXT: load
// CHECK-NEXT: ret i64
// CHECK-NOT: switch
pub fn get_len(arg: &K)->usize{
    match arg {
        K::A(ref lst)=>lst.len(),
        K::B(ref lst)=>lst.len(),
        K::C(ref lst)=>lst.len(),
        K::D(ref lst)=>lst.len(),
    }
}
