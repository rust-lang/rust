struct Ref<'a, T: 'a> {
    data: &'a T
}

fn foo<'a>(x: Ref<i32>, y: &mut Vec<Ref<'a, i32>>) {
    y.push(x); //~ ERROR explicit lifetime
}

fn main() { }
