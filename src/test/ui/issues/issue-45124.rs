// run-pass
#![allow(unreachable_code)]
// compile-flags: --edition 2018

#![feature(try_blocks)]

fn main() {
    let mut a = 0;
    let () = {
        let _: Result<(), ()> = try {
            let _ = Err(())?;
            return
        };
        a += 1;
    };
    a += 2;
    assert_eq!(a, 3);
}
