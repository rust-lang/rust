fn main() {
    let ref my_ref @ _ = 0;
    *my_ref = 0;
    //~^ ERROR cannot assign to `*my_ref`, which is behind an `&` reference [E0594]
}
