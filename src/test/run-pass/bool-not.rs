pub fn main() {
    if !false { assert!((true)); } else { assert!((false)); }
    if !true { assert!((false)); } else { assert!((true)); }
}
