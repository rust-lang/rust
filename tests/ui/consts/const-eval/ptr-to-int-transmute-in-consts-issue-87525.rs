const fn foo(ptr: *const u8) -> usize {
    unsafe {
        std::mem::transmute(ptr)
        //~^ WARN pointers cannot be transmuted to integers
    }
}

trait Human {
    const ID: usize = {
        let value = 10;
        let ptr: *const usize = &value;
        unsafe {
            std::mem::transmute(ptr)
            //~^ WARN pointers cannot be transmuted to integers
        }
    };

    fn id_plus_one() -> usize {
        Self::ID + 1
    }
}

struct Type<T>(T);

impl<T> Type<T> {
    const ID: usize = {
        let value = 10;
        let ptr: *const usize = &value;
        unsafe {
            std::mem::transmute(ptr)
            //~^ WARN pointers cannot be transmuted to integers
        }
    };

    fn id_plus_one() -> usize {
        Self::ID + 1
    }
}

fn control(ptr: *const u8) -> usize {
    unsafe {
        std::mem::transmute(ptr)
    }
}

struct ControlStruct;

impl ControlStruct {
    fn new() -> usize {
        let value = 10;
        let ptr: *const i32 = &value;
        unsafe {
            std::mem::transmute(ptr)
        }
    }
}


const fn zoom(ptr: *const u8) -> usize {
    unsafe {
        std::mem::transmute(ptr)
        //~^ WARN pointers cannot be transmuted to integers
    }
}

fn main() {
    const a: u8 = 10;
    const value: usize = zoom(&a);
    //~^ ERROR evaluation of constant value failed
}
