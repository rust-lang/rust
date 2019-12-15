// Ensures the independence of each side in `binding @ subpat`
// determine their binding modes independently of each other.
//
// That is, `binding` does not influence `subpat`.
// This is important because we might want to allow `p1 @ p2`,
// where both `p1` and `p2` are syntactically unrestricted patterns.
// If `binding` is allowed to influence `subpat`,
// this would create problems for the generalization aforementioned.

#![feature(bindings_after_at)]

fn main() {
    struct NotCopy;

    fn f1(a @ b: &NotCopy) { // OK
        let _: &NotCopy = a;
    }
    fn f2(ref a @ b: &NotCopy) {
        let _: &&NotCopy = a; // Ok
    }

    let a @ b = &NotCopy; // OK
    let _: &NotCopy = a;
    let ref a @ b = &NotCopy; // OK
    let _: &&NotCopy = a;

    let ref a @ b = NotCopy; //~ ERROR cannot bind by-move and by-ref in the same pattern
    let ref mut a @ b = NotCopy; //~ ERROR cannot bind by-move and by-ref in the same pattern
    match Ok(NotCopy) {
        Ok(ref a @ b) | Err(ref a @ b) => {}
        //~^ ERROR cannot bind by-move and by-ref in the same pattern
    }
    match NotCopy {
        ref a @ b => {}
        //~^ ERROR cannot bind by-move and by-ref in the same pattern
    }
}
