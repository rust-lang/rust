#![deny(dead_code)]

enum Category {
    Dead, //~ ERROR variant `Dead` is never constructed
    Used,
}

trait UnusedTrait { //~ ERROR trait `UnusedTrait` is never used
    fn this_is_unused(&self) -> Category {
        Category::Dead
    }
}

struct UnusedStruct; //~ ERROR struct `UnusedStruct` is never constructed

impl UnusedTrait for UnusedStruct {
    fn this_is_unused(&self) -> Category {
        Category::Used
    }
}

mod private {
    #[derive(Debug)]
    struct UnusedStruct; //~ ERROR struct `UnusedStruct` is never constructed
}

fn main() {
    let _c = Category::Used;
}
