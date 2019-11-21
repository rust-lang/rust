fn main() {
    foo(&mut 5);
}

const fn foo(x: &mut i32) -> i32 { //~ ERROR mutable references in const fn are unstable
    *x + 1
}
