use std::cell::RefCell;

fn lots_of_funny_borrows() {
    let c = RefCell::new(42);
    {
        let s1 = c.borrow();
        let _x: i32 = *s1;
        let s2 = c.borrow();
        let _x: i32 = *s1;
        let _y: i32 = *s2;
        let _x: i32 = *s1;
        let _y: i32 = *s2;
    }
    {
        let mut m = c.borrow_mut();
        let _z: i32 = *m;
        {
            let s: &i32 = &*m;
            let _x = *s;
        }
        *m = 23;
        let _z: i32 = *m;
    }
    {
        let s1 = c.borrow();
        let _x: i32 = *s1;
        let s2 = c.borrow();
        let _x: i32 = *s1;
        let _y: i32 = *s2;
        let _x: i32 = *s1;
        let _y: i32 = *s2;
    }
}

fn main() {
    lots_of_funny_borrows();
}
