// compile-flags: -Z parse-only

fn foo(_: *()) {
    //~^ expected mut or const in raw pointer type (use `*mut T` or `*const T` as appropriate)
}
