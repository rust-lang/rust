fn main() {
    return
        { return () }
//~^ ERROR type annotations needed [E0282]
    ()
    ;
}
