//@check-pass

#![allow(harmful_unused_attributes, reason = "the lint can be silenced")]
#![crate_type = "lib"]

fn foo(){
    #[repr(C)]
    println!("foo");
}
