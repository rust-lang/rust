// Test that macro-expanded non-inline modules behave correctly

macro_rules! mod_decl {
    ($i:ident) => { mod $i; }
}

mod macro_expanded_mod_helper {
    mod_decl!(foo); // This should search in the folder `macro_expanded_mod_helper`
}

fn main() {
    mod_decl!(foo);
    //~^ ERROR Cannot declare a non-inline module inside a block
}
