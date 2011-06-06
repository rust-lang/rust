// error-pattern:can not create alias

fn foo(&int x) {
    log x;
}

fn main() {
    auto box = @mutable 1;
    foo(*box);
}
