enum TestEnum {
    Item(i32),
}

fn test(_: &mut i32) {
}

fn main() {
    let mut x = TestEnum::Item(10);
    match x {
        TestEnum::Item(ref mut x) => {
            test(&mut x); //~ ERROR cannot borrow `x` as mutable, as it is not declared as mutable
            //~| HELP try removing `&mut` here
        }
    }
}
