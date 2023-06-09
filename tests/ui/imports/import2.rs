use baz::zed::bar; //~ ERROR unresolved import `baz::zed` [E0432]
                   //~^ could not find `zed` in `baz`

mod baz {}
mod zed {
    pub fn bar() { println!("bar3"); }
}
fn main() {
    bar();
}
