fn main() {
    fn x(a: &char) {
        let &b = a;
        b.make_ascii_uppercase();
//~^ cannot borrow `b` as mutable, as it is not declared as mutable
    }
}
