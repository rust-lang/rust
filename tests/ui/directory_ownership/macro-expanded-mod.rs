// Test that macro-expanded file modules behave correctly

macro_rules! mod_decl {
    ($i:ident) => {
        mod $i; //~ ERROR cannot declare a file module inside a block unless it has a path attribute
    };
}

mod macro_expanded_mod_helper {
    mod_decl!(foo); // This should search in the folder `macro_expanded_mod_helper`
}

fn main() {
    mod_decl!(foo);
}
