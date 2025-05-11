//@ only-cdb
//@ compile-flags:-g

// === CDB TESTS ==================================================================================

// cdb-command: g
// cdb-command: dx basic
// cdb-check: basic            : [...] : a!core::ops::function::FnOnce::call_once<fn_ptr::main::closure_env$0,tuple$<i32,i32> >+0x0 [Type: int (__cdecl*)(int,int)]
// cdb-check: a!core::ops::function::FnOnce::call_once<fn_ptr::main::closure_env$0,tuple$<i32,i32> >+0x0 [Type: int __cdecl(int,int)]

// cdb-command: dx paramless
// cdb-check: paramless        : [...] : a!core::ops::function::FnOnce::call_once<fn_ptr::main::closure_env$1,tuple$<> >+0x0 [Type: int (__cdecl*)()]
// cdb-check: a!core::ops::function::FnOnce::call_once<fn_ptr::main::closure_env$1,tuple$<> >+0x0 [Type: int __cdecl()]

// cdb-command: dx my_struct
// cdb-check: my_struct        [Type: fn_ptr::MyStruct]
// cdb-check:   [+0x000] my_field         : [...] : a!core::ops::function::FnOnce::call_once<fn_ptr::main::closure_env$2,tuple$<ref$<fn_ptr::MyStruct> > >+0x0 [Type: int (__cdecl*)(fn_ptr::MyStruct *)]

// cdb-command: dx non_rec_struct
// cdb-check: non_rec_struct   [Type: fn_ptr::NonRecStruct]
// cdb-check:  [+0x000] my_field         : [...] : a!core::ops::function::FnOnce::call_once<fn_ptr::main::closure_env$3,tuple$<i32> >+0x0 [Type: int (__cdecl*)(int)]

type BasicFnPtr = fn(i32, i32) -> i32;

pub type ParamlessFnPtr = fn() -> i32;

type MyFnPtr = fn(b: &MyStruct) -> i32;

type NonRecFnPtr = fn(i: i32) -> i32;

struct MyStruct {
    my_field: MyFnPtr,
}

struct NonRecStruct {
    my_field: NonRecFnPtr,
}

fn main() {
    let basic: BasicFnPtr = |a, b| a + b;
    let paramless: ParamlessFnPtr = || 1;
    let my_struct = MyStruct { my_field: |_| 1 };
    let non_rec_struct = NonRecStruct { my_field: |i| i };

    _zzz(); // #break
}

#[inline(never)]
fn _zzz() {
    ()
}
