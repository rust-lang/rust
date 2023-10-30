use autodiff::autodiff;

#[autodiff(d_sum, Forward, Duplicated)]
fn sum(#[dup] x: &Vec<Vec<f32>>) -> f32 {
    x.into_iter().map(|x| x.into_iter().map(|x| x.sqrt())).flatten().sum()
}

fn main() {
    let a = vec![vec![1.0, 2.0, 4.0, 8.0]];
    //let mut b = vec![vec![0.0, 0.0, 0.0, 0.0]];
    let b = vec![vec![1.0, 0.0, 0.0, 0.0]];

    dbg!(&d_sum(&a, &b));

    dbg!(&b);
}

#[cfg(test)]
mod tests {
    #[test]
    fn main() {
        super::main()
    }
}
