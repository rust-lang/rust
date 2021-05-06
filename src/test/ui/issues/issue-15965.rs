fn main() {
    return
        { return () }
//~^ ERROR expected function, found `_`
    ()
    ;
}
