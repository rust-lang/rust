fn f(_: B) {}
//~^ ERROR cannot find type `B` in this scope [E0412]

fn main() {
    let _ = [0; f as usize];
}
