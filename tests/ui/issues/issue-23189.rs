mod module {}

fn main() {
    let _ = module { x: 0 }; //~ERROR expected struct
}
