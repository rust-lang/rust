struct X { x: i32 }

fn main() {
    let mut b: Vec<X> = vec![];
    b.sort();
    //~^ ERROR `X: std::cmp::Ord` is not satisfied
}
