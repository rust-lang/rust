//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: cannot transmute_copy if Dst is larger than Src

fn main() {
    unsafe {
        let _unused: u64 = std::mem::transmute_copy(&1_u8);
    }
}
