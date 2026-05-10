//@compile-flags: --test

#![warn(clippy::unwrap_used)]
#![warn(clippy::get_unwrap)]

fn main() {}

#[test]
fn test() {
    let boxed_slice: Box<[u8]> = Box::new([0, 1, 2, 3]);
    let _ = boxed_slice.get(1).unwrap();
    //~^ get_unwrap
}

#[cfg(test)]
mod issue9612 {
    // should not lint in `#[cfg(test)]` modules
    #[test]
    fn test_fn() {
        let _a: u8 = 2.try_into().unwrap();
        let _a: u8 = 3.try_into().expect("");

        util();
    }

    #[allow(unconditional_panic)]
    fn util() {
        let _a: u8 = 4.try_into().unwrap();
        let _a: u8 = 5.try_into().expect("");
        // should still warn
        let _ = Box::new([0]).get(1).unwrap();
        //~^ get_unwrap
    }
}
