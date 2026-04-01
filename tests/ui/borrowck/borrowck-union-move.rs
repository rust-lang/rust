use std::mem::ManuallyDrop;

#[derive(Clone, Copy)]
struct Copy;
struct NonCopy;

union Unn {
    n1: ManuallyDrop<NonCopy>,
    n2: ManuallyDrop<NonCopy>,
}
union Ucc {
    c1: Copy,
    c2: Copy,
}
union Ucn {
    c: Copy,
    n: ManuallyDrop<NonCopy>,
}

fn main() {
    unsafe {
        // 2 NonCopy
        {
            let mut u = Unn { n1: ManuallyDrop::new(NonCopy) };
            let a = u.n1;
            let a = u.n1; //~ ERROR use of moved value: `u`
        }
        {
            let mut u = Unn { n1: ManuallyDrop::new(NonCopy) };
            let a = u.n1;
            let a = u; //~ ERROR use of moved value: `u`
        }
        {
            let mut u = Unn { n1: ManuallyDrop::new(NonCopy) };
            let a = u.n1;
            let a = u.n2; //~ ERROR use of moved value: `u`
        }
        // 2 Copy
        {
            let mut u = Ucc { c1: Copy };
            let a = u.c1;
            let a = u.c1; // OK
        }
        {
            let mut u = Ucc { c1: Copy };
            let a = u.c1;
            let a = u; // OK
        }
        {
            let mut u = Ucc { c1: Copy };
            let a = u.c1;
            let a = u.c2; // OK
        }
        // 1 Copy, 1 NonCopy
        {
            let mut u = Ucn { c: Copy };
            let a = u.c;
            let a = u.c; // OK
        }
        {
            let mut u = Ucn { c: Copy };
            let a = u.n;
            let a = u.n; //~ ERROR use of moved value: `u`
        }
        {
            let mut u = Ucn { c: Copy };
            let a = u.n;
            let a = u.c; //~ ERROR use of moved value: `u`
        }
        {
            let mut u = Ucn { c: Copy };
            let a = u.c;
            let a = u.n; // OK
        }
        {
            let mut u = Ucn { c: Copy };
            let a = u.c;
            let a = u; // OK
        }
        {
            let mut u = Ucn { c: Copy };
            let a = u.n;
            let a = u; //~ ERROR use of moved value: `u`
        }
    }
}
