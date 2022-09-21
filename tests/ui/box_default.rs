#![warn(clippy::box_default)]

#[derive(Default)]
struct ImplementsDefault;

struct OwnDefault;

impl OwnDefault {
    fn default() -> Self {
        Self
    }
}

macro_rules! outer {
    ($e: expr) => {
        $e
    };
}

fn main() {
    let _string: Box<String> = Box::new(Default::default());
    let _byte = Box::new(u8::default());
    let _vec = Box::new(Vec::<u8>::new());
    let _impl = Box::new(ImplementsDefault::default());
    let _impl2 = Box::new(<ImplementsDefault as Default>::default());
    let _impl3: Box<ImplementsDefault> = Box::new(Default::default());
    let _own = Box::new(OwnDefault::default()); // should not lint
    let _in_macro = outer!(Box::new(String::new()));
    // false negative: default is from different expansion
    let _vec2: Box<Vec<ImplementsDefault>> = Box::new(vec![]);
}
