struct X { x: (); drop { error!("destructor runs"); } }

enum double_option<T,U> { some2(T,U), none2 }

fn main() {
    let x = some2(X { x: () }, X { x: () });
    match move x {
        some2(ref _y, move _z) => { }, //~ ERROR cannot bind by-move and by-ref in the same pattern
        none2 => fail
    }
}
