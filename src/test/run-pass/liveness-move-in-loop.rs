fn take(-x: int) -> int {x}

fn the_loop() {
    let mut list = [];
    loop {
        let x = 5;
        if x > 3 {
            list += [take(x)];
        } else {
            break;
        }
    }
}

fn main() {}
