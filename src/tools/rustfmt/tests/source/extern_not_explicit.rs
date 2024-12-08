// rustfmt-force_explicit_abi: false

 extern  "C" {
   fn some_fn() -> ();
 }

 extern "C" fn sup() {

 }

type funky_func = extern "C" fn (unsafe extern "rust-call" fn(*const JSJitInfo, *mut JSContext,
                                              HandleObject, *mut libc::c_void, u32,
                                              *mut JSVal)
                                              -> u8);
