#![deny(dangling_pointers_from_temporaries)]

fn main() {
    vec![0u8].as_ptr();
    //~^ ERROR getting a pointer from a temporary `Vec<u8>` will result in a dangling pointer
    vec![0u8].as_mut_ptr();
    //~^ ERROR getting a pointer from a temporary `Vec<u8>` will result in a dangling pointer
}
