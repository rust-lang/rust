use a::f;
use b::f; //~ ERROR: unresolved import `b::f` [E0432]
          //~^ no `f` in `b`

mod a { pub fn f() {} }
mod b { }

fn main() {
    f();
}
