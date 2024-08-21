const FOO: () = {
    let x = std::cell::Cell::new(42);
    let y = &x; //~ERROR: cannot borrow here
};

const FOO2: () = {
    let mut x = std::cell::Cell::new(42);
    let y = &*&mut x; //~ERROR: cannot borrow here
};

const FOO3: () = unsafe {
    let mut x = std::cell::Cell::new(42);
    let y = &*(&mut x as *mut _); //~ERROR: cannot borrow here
};

fn main() {}
