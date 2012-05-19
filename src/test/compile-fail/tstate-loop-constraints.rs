pure fn is_even(i: int) -> bool { (i%2) == 0 }
fn even(i: int) : is_even(i) -> int { i }

fn main() {

    let mut x: int = 42;
    loop {
        loop {
            loop {
                check is_even(x);
                even(x); // OK
                loop {
                    even(x); //! ERROR unsatisfied precondition
                    x = 11; 
                }
            }
        }
    }
}
