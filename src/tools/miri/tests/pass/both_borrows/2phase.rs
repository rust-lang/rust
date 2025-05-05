//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows

trait S: Sized {
    fn tpb(&mut self, _s: Self) {}
}

impl S for i32 {}

fn two_phase1() {
    let mut x = 3;
    x.tpb(x);
}

fn two_phase2() {
    let mut v = vec![];
    v.push(v.len());
}

fn two_phase3(b: bool) {
    let mut x = &mut vec![];
    let mut y = vec![];
    x.push((
        {
            if b {
                x = &mut y;
            }
            22
        },
        x.len(),
    ));
}

fn two_phase_raw() {
    let x: &mut Vec<i32> = &mut vec![];
    #[allow(unreachable_code)] // The `push` itself never gets reached.
    x.push({
        // Unfortunately this does not trigger the problem of creating a
        // raw ponter from a pointer that had a two-phase borrow derived from
        // it because of the implicit &mut reborrow.
        let raw = x as *mut _;
        unsafe {
            *raw = vec![1];
        }
        return;
    });
}

fn two_phase_overlapping1() {
    let mut x = vec![];
    let p = &x;
    x.push(p.len());
}

fn two_phase_overlapping2() {
    use std::ops::AddAssign;
    let mut x = 1;
    let l = &x;
    x.add_assign(x + *l);
}

fn main() {
    two_phase1();
    two_phase2();
    two_phase3(false);
    two_phase3(true);
    two_phase_raw();
    two_phase_overlapping1();
    two_phase_overlapping2();
}
