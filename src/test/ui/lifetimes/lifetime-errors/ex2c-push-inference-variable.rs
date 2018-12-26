struct Ref<'a, T: 'a> {
    data: &'a T
}

fn foo<'a, 'b, 'c>(x: &'a mut Vec<Ref<'b, i32>>, y: Ref<'c, i32>) {
    let z = Ref { data: y.data };
    x.push(z); //~ ERROR lifetime mismatch
}

fn main() { }
