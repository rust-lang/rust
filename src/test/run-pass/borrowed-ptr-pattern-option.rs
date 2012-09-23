fn select(x: &r/Option<int>, y: &r/Option<int>) -> &r/Option<int> {
    match (x, y) {
        (&None, &None) => x,
        (&Some(_), _) => x,
        (&None, &Some(_)) => y
    }
}

fn main() {
    let x = None;
    let y = Some(3);
    assert select(&x, &y).get() == 3;
}