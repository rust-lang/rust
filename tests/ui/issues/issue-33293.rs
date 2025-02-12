fn main() {
    match 0 {
        aaa::bbb(_) => ()
        //~^ ERROR failed to resolve: use of unresolved module or unlinked crate `aaa`
    };
}
