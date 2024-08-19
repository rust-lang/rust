#![deny(dangling_pointers_from_temporaries)]

fn main() {
    vec![0u8].as_ptr(); //~ ERROR [dangling_pointers_from_temporaries]
    vec![0u8].as_mut_ptr(); //~ ERROR [dangling_pointers_from_temporaries]
}
