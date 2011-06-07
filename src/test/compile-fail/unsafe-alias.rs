// error-pattern:may alias with argument

fn foo(&int x, fn() f) {
    log x;
}

fn whoknows(@mutable int x) {
    *x = 10;
}

fn main() {
    auto box = @mutable 1;
    foo(*box, bind whoknows(box));
}
