struct Wrap<'a> { w: &'a mut u32 }

fn foo() {
    let mut x = 22;
    let wrapper = Wrap { w: &mut x };
    x += 1; //~ ERROR cannot use `x` because it was mutably borrowed [E0503]
    *wrapper.w += 1;
}

fn main() { }
