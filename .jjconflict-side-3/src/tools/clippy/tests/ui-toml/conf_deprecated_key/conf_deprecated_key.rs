//@no-rustfix
//@error-in-other-file: use of a deprecated field
//@error-in-other-file: use of a deprecated field

fn main() {}

#[warn(clippy::cognitive_complexity)]
fn cognitive_complexity() {
    //~^ cognitive_complexity
    let x = vec![1, 2, 3];
    for i in x {
        if i == 1 {
            println!("{i}");
        }
    }
}
