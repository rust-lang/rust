fn main() {
    #[clippy::author]
    let print_text = |x| println!("{}", x);
    print_text("hello");
}
