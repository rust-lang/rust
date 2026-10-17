// Ensures the independence of each side in `binding @ subpat`
// determine their binding modes independently of each other.
//
// That is, `binding` does not influence `subpat`.
// This is important because we might want to allow `p1 @ p2`,
// where both `p1` and `p2` are syntactically unrestricted patterns.
// If `binding` is allowed to influence `subpat`,
// this would create problems for the generalization aforementioned.


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

    let ref a @ b = NotCopy; //~ ERROR cannot move out of value because it is borrowed
    let _a: &NotCopy = a;
    let _b: NotCopy = b;
    let ref mut a @ b = NotCopy; //~ ERROR cannot move out of value because it is borrowed
    //~^ ERROR borrow of moved value
    let _a: &NotCopy = a;
    let _b: NotCopy = b;
    match Ok(NotCopy) {
        Ok(ref a @ b) | Err(b @ ref a) => {
            //~^ ERROR cannot move out of value because it is borrowed
            //~| ERROR borrow of moved value
            let _a: &NotCopy = a;
            let _b: NotCopy = b;
        }
    }
    match NotCopy {
        ref a @ b => {
            //~^ ERROR cannot move out of value because it is borrowed
            let _a: &NotCopy = a;
            let _b: NotCopy = b;
        }
    }
}
