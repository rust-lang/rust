//@ run-rustfix

struct _S(u32, Vec<i32>);

fn _foo(x: &_S) {
    match x {
        _S(& (mut _y), _v) => {
        //~^ ERROR mismatched types [E0308]
        }
    }
}

fn main() {
}
