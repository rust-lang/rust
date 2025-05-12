// Test that assignments to an `&mut` pointer which is found in a
// borrowed (but otherwise non-aliasable) location is illegal.

struct S<'a> {
    pointer: &'a mut isize
}

fn copy_borrowed_ptr<'a>(p: &'a mut S<'a>) -> S<'a> {
    S { pointer: &mut *p.pointer }
}

fn main() {
    let mut x = 1;

    {
        let mut y = S { pointer: &mut x };
        let z = copy_borrowed_ptr(&mut y);
        *y.pointer += 1;
        //~^ ERROR cannot use `*y.pointer`
        //~| ERROR cannot assign to `*y.pointer`
        *z.pointer += 1;
    }
}
