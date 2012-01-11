fn foo(x: fn()) {
    bind x(); //! ERROR cannot bind fn closures
}

fn bar(x: fn&()) {
    bind x(); //! ERROR cannot bind fn& closures
}

fn main() {
}
