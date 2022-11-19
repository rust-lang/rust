//@ run-pass
//@ compile-flags: -Zmir-opt-level=0 -Cdebug-assertions=no -Zub-checks=yes

fn main() {
    for val in i8::MIN..=i8::MAX {
        if val != 0 {
            let x = std::num::NonZeroI8::new(val).unwrap();
            if val != i8::MIN {
                let _y = -x;
            }
        }
    }
}
