// Check that `need_type_info` distinguishes between a single placeholder and
// multiple placeholders when suggesting an explicit type.

fn singular() {
    let v = &[];
    //~^ ERROR type annotations needed
    let _ = v.iter();
}

fn plural() {
    let x = (vec![], vec![]);
    //~^ ERROR type annotations needed
    let _ = x;
}

fn main() {}
