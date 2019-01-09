fn main() {
    assert_eq!(1, 2)
    assert_eq!(3, 4) //~ ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `assert_eq`
    println!("hello");
}
