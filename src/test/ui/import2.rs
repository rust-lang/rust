use baz::zed::bar; //~ ERROR unresolved import `baz::zed` [E0432]
                   //~^ Could not find `zed` in `baz`

mod baz {}
mod zed {
    pub fn bar() { println!("bar3"); }
}
fn main() {
    bar();
}
