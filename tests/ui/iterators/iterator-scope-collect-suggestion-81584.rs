// https://github.com/rust-lang/rust/issues/81584
//@ run-rustfix
fn main() {
        let _ = vec![vec![0, 1], vec![2]]
            .into_iter()
            .map(|y| y.iter().map(|x| x + 1))
                  //~^ ERROR cannot return value referencing function parameter `y`
            .collect::<Vec<_>>();
}
