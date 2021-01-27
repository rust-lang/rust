#![warn(clippy::for_loops_over_options_or_results)]

fn main() {
    let x = vec![Some(1), Some(2), Some(3)];
    for n in x {
        if let Some(n) = n {
            println!("{}", n);
        }
    }

    let y: Vec<Result<i32, i32>> = vec![];
    for n in y.clone() {
        if let Ok(n) = n {
            println!("{}", n);
        }
    }

    // This should not trigger the lint
    for n in y.clone() {
        if let Ok(n) = n {
            println!("{}", n);
        } else {
            println!("Oops!");
        }
    }

    // This should not trigger the lint
    for n in vec![Some(1), Some(2), Some(3)].iter().flatten() {
        println!("{}", n);
    }
}
