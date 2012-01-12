// error-pattern:Unsatisfied precondition constraint

fn main() {
    let x = 5;
    let _y = fn~[move x]() { };
    let _z = x; //< error: Unsatisfied precondition constraint
}
