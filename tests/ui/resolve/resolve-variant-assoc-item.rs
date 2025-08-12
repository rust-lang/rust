enum E { V }
use E::V;

fn main() {
    E::V::associated_item; //~ ERROR: cannot find
    V::associated_item; //~ ERROR: cannot find
}
