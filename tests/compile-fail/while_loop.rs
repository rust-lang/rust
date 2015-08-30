#![feature(plugin)]
#![plugin(clippy)]

#[deny(while_let_loop)]
fn main() {
    let y = Some(true);
    loop { //~ERROR
        if let Some(_x) = y {
            let _v = 1;
        } else {
            break;
        }
    }
    loop { //~ERROR
        if let Some(_x) = y {
            let _v = 1;
        } else {
            break
        }
    }
    loop { // no error, break is not in else clause
        if let Some(_x) = y {
            let _v = 1;
        }
        break;
    }
    loop { //~ERROR
        match y {
            Some(_x) => true,
            None => break
        };
    }
    loop { // no error, match is not the only statement
        match y {
            Some(_x) => true,
            None => break
        };
        let _x = 1;
    }
    loop { // no error, else branch does something other than break
        match y {
            Some(_x) => true,
            _ => {
                let _z = 1;
                break;
            }
        };
    }
    while let Some(x) = y { // no error, obviously
        println!("{}", x);
    }
}
