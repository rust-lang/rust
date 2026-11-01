//@ check-pass

fn main() {
    let _my_usize = const {
        let a = 10_usize;
        let b: &'_ usize = &a;
        *b
    };
}
