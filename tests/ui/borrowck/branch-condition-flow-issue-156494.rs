//@ edition: 2024

fn main() {
    let v1: Vec<i32> = vec![1, 2, 3];
    let mut iter = v1.iter();

    let mut first_encounter = true;
    let mut token: i32;
    let mut following_token: i32;
    loop {
        if first_encounter {
            if let Some(token) = iter.next() {
                match iter.next() {
                    Some(temp) => {
                        following_token = *temp;
                        first_encounter = false;
                        println!("{} followed by {}", token, following_token);
                    }
                    _ => {
                        println!("{} is last", token);
                    }
                }
            } else {
                break;
            }
        } else {
            token = following_token; //~ ERROR used binding `following_token` isn't initialized
            first_encounter = true;
            match iter.next() {
                Some(temp) => {
                    following_token = *temp;
                    first_encounter = false;
                    println!("{} followed by {}", token, following_token);
                }
                _ => {
                    println!("{} is last", token);
                }
            }
        }
    }
}

fn guarded_match(value: i32, condition: bool) {
    let y;
    match value {
        0 => {
            y = 1;
        }
        _ if condition => {}
        _ => {
            y = 2;
        }
    }
    let _z = y; //~ ERROR used binding `y` is possibly-uninitialized
}
