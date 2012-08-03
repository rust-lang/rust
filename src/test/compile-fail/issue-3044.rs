// error-pattern: Non-function passed to a `do` function as its last argument, or wrong number of arguments passed to a `do` function
fn main() {
    let needlesArr: ~[char] = ~['a', 'f'];
    do vec::foldr(needlesArr) |x, y| {
    }
// for some reason if I use the new error syntax for the two error messages this generates,
// the test runner gets confused -- tjc
}

