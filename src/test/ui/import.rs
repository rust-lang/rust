use zed::bar;
use zed::baz; //~ ERROR unresolved import `zed::baz` [E0432]
              //~^ no `baz` in `zed`. Did you mean to use `bar`?


mod zed {
    pub fn bar() { println!("bar"); }
    use foo; //~ ERROR unresolved import `foo` [E0432]
             //~^ no `foo` in the root
}

fn main() {
    zed::foo(); //~ ERROR `foo` is private
    bar();
}
