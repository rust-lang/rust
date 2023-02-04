// check-pass
#![feature(const_for)]
#![feature(const_mut_refs)]
#![feature(const_trait_impl)]
#![feature(const_iter)]
#![feature(const_intoiterator_identity)]
#![feature(inline_const)]

fn main() {
    const {
        let mut arr = [0; 3];
        for i in 0..arr.len() {
            arr[i] = i;
        }
        assert!(arr[0] == 0);
        assert!(arr[1] == 1);
        assert!(arr[2] == 2);
    }
}
