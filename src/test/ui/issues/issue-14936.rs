// build-pass
#![allow(unused_macros)]
#![allow(dead_code)]
#![feature(llvm_asm)]

type History = Vec<&'static str>;

fn wrap<A>(x:A, which: &'static str, history: &mut History) -> A {
    history.push(which);
    x
}

macro_rules! demo {
    ( $output_constraint:tt ) => {
        {
            let mut x: isize = 0;
            let y: isize = 1;

            let mut history: History = vec![];
            unsafe {
                llvm_asm!("mov ($1), $0"
                          : $output_constraint (*wrap(&mut x, "out", &mut history))
                          : "r"(&wrap(y, "in", &mut history))
                          :: "volatile");
            }
            assert_eq!((x,y), (1,1));
            let b: &[_] = &["out", "in"];
            assert_eq!(history, b);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn main() {
    fn out_write_only_expr_then_in_expr() {
        demo!("=r")
    }

    fn out_read_write_expr_then_in_expr() {
        demo!("+r")
    }

    out_write_only_expr_then_in_expr();
    out_read_write_expr_then_in_expr();
}

#[cfg(all(not(target_arch = "x86"), not(target_arch = "x86_64")))]
pub fn main() {}
