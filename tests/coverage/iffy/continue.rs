#![allow(unused_assignments, unused_variables)]

fn main() {
    let is_true = std::env::args().len() == 1;

    let mut x = 0;
    for _ in 0..10 {
        match is_true {
            true => {
                continue;
            }
            _ => {
                x = 1;
            }
        }
        x = 3;
    }
    for _ in 0..10 {
        match is_true {
            false => {
                x = 1;
            }
            _ => {
                continue;
            }
        }
        x = 3;
    }
    for _ in 0..10 {
        match is_true {
            true => {
                x = 1;
            }
            _ => {
                continue;
            }
        }
        x = 3;
    }
    for _ in 0..10 {
        if is_true {
            continue;
        }
        x = 3;
    }
    for _ in 0..10 {
        match is_true {
            false => {
                x = 1;
            }
            _ => {
                let _ = x;
            }
        }
        x = 3;
    }
    for _ in 0..10 {
        match is_true {
            false => {
                x = 1;
            }
            _ => {
                break;
            }
        }
        x = 3;
    }
    let _ = x;
}
