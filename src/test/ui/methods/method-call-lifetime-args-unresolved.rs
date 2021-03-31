fn main() {
    0.clone::<'a>(); //~ ERROR use of undeclared lifetime name `'a`
}
