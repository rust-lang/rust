mod foo { pub fn bar() {} }

fn main() {
    match () {
        foo::bar => {} //~ ERROR expected unit struct/variant or constant, found function `foo::bar`
    }
}
