//@ run-pass

// Check that when a `let`-binding occurs in a loop, its associated
// drop-flag is reinitialized (to indicate "needs-drop" at the end of
// the owning variable's scope).

struct A<'a>(&'a mut i32);

impl<'a> Drop for A<'a> {
    fn drop(&mut self) {
        *self.0 += 1;
    }
}

fn main() {
    let mut cnt = 0;
    for i in 0..2 {
        let a = A(&mut cnt);
        if i == 1 { // Note that
            break;  //  both this break
        }           //   and also
        drop(a);    //    this move of `a`
        // are necessary to expose the bug
    }
    assert_eq!(cnt, 2);
}
