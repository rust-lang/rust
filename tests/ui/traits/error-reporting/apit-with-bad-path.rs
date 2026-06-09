// Ensure that we don't emit an E0270 for "`impl AsRef<Path>: AsRef<Path>` not satisfied".

fn foo(filename: impl AsRef<Path>) {
    //~^ ERROR cannot find type `Path` in this scope
    std::fs::write(filename, "hello").unwrap();
}

fn main() {
    foo("/tmp/hello");
}
