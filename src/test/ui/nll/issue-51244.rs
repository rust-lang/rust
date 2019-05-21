fn main() {
    let ref my_ref @ _ = 0;
    *my_ref = 0;
    //~^ ERROR cannot assign to `*my_ref` which is behind a `&` reference [E0594]
}
