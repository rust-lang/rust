fn main() {
    // Unconstrained type:
    format!("{:?}", None);
    //~^ ERROR type annotations needed [E0282]
}
