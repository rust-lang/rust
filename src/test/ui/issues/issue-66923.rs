fn main() {
    let v = vec![1_f64, 2.2_f64];
    let mut fft: Vec<Vec<f64>> = vec![];

    let x1: &[f64] = &v;
    let x2: Vec<f64> = x1.into_iter().collect();
    //~^ ERROR a collection of type
    fft.push(x2);

    let x3 = x1.into_iter().collect::<Vec<f64>>();
    //~^ ERROR a collection of type
    fft.push(x3);
}
