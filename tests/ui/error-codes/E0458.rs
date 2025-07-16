#[link(kind = "wonderful_unicorn")] extern "C" {} //~ ERROR malformed `link` attribute input [E0539]
                                                  //~| ERROR E0459

fn main() {
}
