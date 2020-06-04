fn main() -> u8 { // this will lower to HIR but will not compile: `main` can only return types that implement `std::process::Termination`
    let mut countdown = 10;
    while countdown > 0 {
        if false {
            return if countdown > 8 { 1 } else { return 2; };
        }
        countdown -= 1;
    }
    0
}