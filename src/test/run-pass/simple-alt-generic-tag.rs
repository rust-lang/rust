

enum opt<T> { none, }

fn main() {
    let x = none::<int>;
    match x { none::<int> => { debug!{"hello world"}; } }
}
