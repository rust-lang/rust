enum E { V }
use E::V;

fn main() {
    E::V::associated_item; //~ ERROR failed to resolve: not a module `V`
    V::associated_item; //~ ERROR failed to resolve: not a module `V`
}
