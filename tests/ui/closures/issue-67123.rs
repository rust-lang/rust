fn foo<T>(t: T) {
    || { t; t; }; //~ ERROR: use of moved value
}

fn main() {}
