struct X { x: (), }

impl Drop for X {
    fn drop(&mut self) {
        println!("destructor runs");
    }
}

enum DoubleOption<T,U> { Some2(T,U), None2 }

fn main() {
    let x = DoubleOption::Some2(X { x: () }, X { x: () });
    match x {
        DoubleOption::Some2(ref _y, _z) => { },
        //~^ ERROR cannot bind by-move and by-ref in the same pattern
        DoubleOption::None2 => panic!()
    }
}
