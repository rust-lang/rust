enum E {
//~^ ERROR recursive type `E` has infinite size
    V(E),
}
