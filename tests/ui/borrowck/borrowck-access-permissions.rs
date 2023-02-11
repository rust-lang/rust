static static_x : i32 = 1;
static mut static_x_mut : i32 = 1;

fn main() {
    let x = 1;
    let mut x_mut = 1;

    { // borrow of local
        let _y1 = &mut x; //~ ERROR [E0596]
        let _y2 = &mut x_mut; // No error
    }

    { // borrow of static
        let _y1 = &mut static_x; //~ ERROR [E0596]
        unsafe { let _y2 = &mut static_x_mut; } // No error
    }

    { // borrow of deref to box
        let box_x = Box::new(1);
        let mut box_x_mut = Box::new(1);

        let _y1 = &mut *box_x; //~ ERROR [E0596]
        let _y2 = &mut *box_x_mut; // No error
    }

    { // borrow of deref to reference
        let ref_x = &x;
        let ref_x_mut = &mut x_mut;

        let _y1 = &mut *ref_x; //~ ERROR [E0596]
        let _y2 = &mut *ref_x_mut; // No error
    }

    { // borrow of deref to pointer
        let ptr_x : *const _ = &x;
        let ptr_mut_x : *mut _ = &mut x_mut;

        unsafe {
            let _y1 = &mut *ptr_x; //~ ERROR [E0596]
            let _y2 = &mut *ptr_mut_x; // No error
        }
    }

    { // borrowing mutably through an immutable reference
        struct Foo<'a> { f: &'a mut i32, g: &'a i32 };
        let mut foo = Foo { f: &mut x_mut, g: &x };
        let foo_ref = &foo;
        let _y = &mut *foo_ref.f; //~ ERROR [E0596]
    }
}
