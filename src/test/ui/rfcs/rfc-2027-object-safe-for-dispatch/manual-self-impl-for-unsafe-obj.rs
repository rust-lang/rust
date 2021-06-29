// Check that we can manually implement an object-unsafe trait for its trait object.

// run-pass

#![feature(object_safe_for_dispatch)]

trait Bad {
    fn stat() -> char {
        'A'
    }
    fn virt(&self) -> char {
        'B'
    }
    fn indirect(&self) -> char {
        Self::stat()
    }
}

trait Good {
    fn good_virt(&self) -> char {
        panic!()
    }
    fn good_indirect(&self) -> char {
        panic!()
    }
}

impl<'a> Bad for dyn Bad + 'a {
    fn stat() -> char {
        'C'
    }
    fn virt(&self) -> char {
        'D'
    }
}

struct Struct {}

impl Bad for Struct {}

impl Good for Struct {}

fn main() {
    let s = Struct {};

    let mut res = String::new();

    // Directly call static.
    res.push(Struct::stat()); // "A"
    res.push(<dyn Bad>::stat()); // "AC"

    let good: &dyn Good = &s;

    // These look similar enough...
    let bad = unsafe { std::mem::transmute::<&dyn Good, &dyn Bad>(good) };

    // Call virtual.
    res.push(s.virt()); // "ACB"
    res.push(bad.virt()); // "ACBD"

    // Indirectly call static.
    res.push(s.indirect()); // "ACBDA"
    res.push(bad.indirect()); // "ACBDAC"

    assert_eq!(&res, "ACBDAC");
}
