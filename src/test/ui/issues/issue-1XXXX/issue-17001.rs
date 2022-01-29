mod foo {}

fn main() {
    let p = foo { x: () }; //~ ERROR expected struct, variant or union type, found module `foo`
}
