// error-pattern:explicit panic

fn main() {
    let _x = match true {
        false => 0,
        true => panic!(),
    };
}
