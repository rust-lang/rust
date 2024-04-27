enum E { V }
use E::V;

fn main() {
    E::V::associated_item; //~ ERROR failed to resolve: `V` is a variant, not a module
    V::associated_item; //~ ERROR failed to resolve: `V` is a variant, not a module
}
