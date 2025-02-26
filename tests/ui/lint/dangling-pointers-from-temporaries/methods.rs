#![deny(dangling_pointers_from_temporaries)]

fn main() {
    vec![0u8].as_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `Vec<u8>` will be dropped
    vec![0u8].as_mut_ptr();
    //~^ ERROR a dangling pointer will be produced because the temporary `Vec<u8>` will be dropped
}
