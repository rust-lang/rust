struct X { x: (), }

impl Drop for X {
    fn drop(&mut self) {
        println!("destructor runs");
    }
}

enum double_option<T,U> { some2(T,U), none2 }

fn main() {
    let x = double_option::some2(X { x: () }, X { x: () });
    match x {
        double_option::some2(ref _y, _z) => { },
        //~^ ERROR cannot bind by-move and by-ref in the same pattern
        double_option::none2 => panic!()
    }
}
