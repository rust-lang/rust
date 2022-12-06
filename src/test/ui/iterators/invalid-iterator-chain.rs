fn main() {
    let scores = vec![(0, 0)]
        .iter()
        .map(|(a, b)| {
            a + b;
        });
    println!("{}", scores.sum::<i32>()); //~ ERROR E0277
    println!(
        "{}",
        vec![0, 1] //~ ERROR E0277
            .iter()
            .map(|x| x * 2)
            .map(|x| x as f64)
            .map(|x| x as i64)
            .filter(|x| *x > 0)
            .map(|x| { x + 1 })
            .map(|x| { x; })
            .sum::<i32>(),
    );
    println!("{}", vec![0, 1].iter().map(|x| { x; }).sum::<i32>()); //~ ERROR E0277
    println!("{}", vec![(), ()].iter().sum::<i32>()); //~ ERROR E0277
}
