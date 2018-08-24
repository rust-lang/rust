enum foo { foo_(bar) }
struct bar { x: bar }
//~^ ERROR E0072

fn main() {
}
