#![warn(clippy::replace_box)]

fn with_default<T: Default>(b: &mut Box<T>) {
    *b = Box::new(T::default());
    //~^ replace_box
}

fn with_sized<T>(b: &mut Box<T>, t: T) {
    *b = Box::new(t);
    //~^ replace_box
}

fn with_unsized<const N: usize>(b: &mut Box<[u32]>) {
    // No lint for assigning to Box<T> where T: !Default
    *b = Box::new([42; N]);
}

macro_rules! create_default {
    () => {
        Default::default()
    };
}

macro_rules! create_zero_box {
    () => {
        Box::new(0)
    };
}

macro_rules! same {
    ($v:ident) => {
        $v
    };
}

macro_rules! mac {
    (three) => {
        3u32
    };
}

fn main() {
    let mut b = Box::new(1u32);
    b = Default::default();
    //~^ replace_box
    b = Box::default();
    //~^ replace_box

    // No lint for assigning to the storage
    *b = Default::default();
    *b = u32::default();

    // No lint if either LHS or RHS originates in macro
    b = create_default!();
    b = create_zero_box!();
    same!(b) = Default::default();

    b = Box::new(5);
    //~^ replace_box

    b = Box::new(mac!(three));
    //~^ replace_box

    // No lint for assigning to Box<T> where T: !Default
    let mut b = Box::<str>::from("hi".to_string());
    b = Default::default();

    // No lint for late initializations
    #[allow(clippy::needless_late_init)]
    let bb: Box<u32>;
    bb = Default::default();
}
