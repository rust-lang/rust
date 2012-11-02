enum foo = uint;

fn main() {
    let (foo, _) = (2, 3); //~ ERROR declaration of `foo` shadows
}
