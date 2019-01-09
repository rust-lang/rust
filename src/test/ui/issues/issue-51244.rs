fn main() {
    let ref my_ref @ _ = 0;
    *my_ref = 0; //~ ERROR cannot assign to immutable borrowed content `*my_ref` [E0594]
}
