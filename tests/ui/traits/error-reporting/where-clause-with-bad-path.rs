// Ensure that we don't emit an E0270 for "`impl AsRef<Path>: AsRef<Path>` not satisfied".

fn foo<T: AsRef<Path>>(filename: T) {
    //~^ ERROR cannot find type `Path` in this scope
    std::fs::write(filename, "hello").unwrap();
}

fn main() {
    foo("/tmp/hello");
}
