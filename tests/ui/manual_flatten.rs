#![warn(clippy::manual_flatten)]

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

    let z = vec![Some(1), Some(2), Some(3)];
    let z = z.iter();
    for n in z {
        if let Some(n) = n {
            println!("{}", n);
        }
    }

    // Using the `None` variant should not trigger the lint
    let z = vec![Some(1), Some(2), Some(3)];
    for n in z {
        if n.is_none() {
            println!("Nada.");
        }
    }

    // Using the `Err` variant should not trigger the lint
    for n in y.clone() {
        if let Err(e) = n {
            println!("Oops: {}!", e);
        }
    }

    // Having an else clause should not trigger the lint
    for n in y.clone() {
        if let Ok(n) = n {
            println!("{}", n);
        } else {
            println!("Oops!");
        }
    }

    // Using manual flatten should not trigger the lint
    for n in vec![Some(1), Some(2), Some(3)].iter().flatten() {
        println!("{}", n);
    }
}
