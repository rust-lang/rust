
 // EMIT_MIR tls_access.main.runtime-optimized.after.mir
 //@ compile-flags: -Zmir-opt-level=0

 // CHECK-LABEL: fn main(
 // CHECK: [[TLS_READ:_.*]] = &/*tls*/ mut FOO;
 // CHECK: [[BORROW:_.*]] = &(*[[TLS_READ]]);
 // CHECK: [[TLS_WRITE:_.*]] = &/*tls*/ mut FOO;
 // CHECK: (*[[TLS_WRITE]]) = const 42_u8;

 #![feature(thread_local)]

 #[thread_local]
 static mut FOO: u8 = 3;

 fn main() {
     unsafe {
         let a = &FOO;
         FOO = 42;
     }
 }
