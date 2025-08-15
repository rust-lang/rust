//@ check-fail

#![feature(linkage)]

extern "C" {
    #[linkage = "foo"]
    static foo: *const i32;
//~^^ ERROR: malformed `linkage` attribute input [E0539]
}

fn main() {
    println!("{:?}", unsafe { foo });
}
