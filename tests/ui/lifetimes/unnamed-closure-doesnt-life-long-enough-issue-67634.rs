fn main() {
    [0].iter().flat_map(|a| [0].iter().map(|_| &a)); //~ ERROR closure may outlive
}
