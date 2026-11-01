//@ run-pass




fn f() -> isize {
    if true {
        let _s: String = "should not leak".to_string();
        return 1;
    }
    return 0;
}

pub fn main() { f(); }
