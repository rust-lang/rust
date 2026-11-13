#![deny(dangling_pointers_from_temporaries)]
#![feature(vec_as_non_null)]

fn main() {
    vec![0u8].as_ptr();
    //~^ ERROR dangling pointer
    vec![0u8].as_mut_ptr();
    //~^ ERROR dangling pointer
    vec![0u8].as_non_null();
    //~^ ERROR dangling pointer
}
