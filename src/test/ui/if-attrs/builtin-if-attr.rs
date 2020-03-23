// check-pass

fn main() {
    #[allow(unused_variables)]
    if true {
        let a = 1;
    } else if false {
        let b = 1;
    } else {
        let c = 1;
    }
}
