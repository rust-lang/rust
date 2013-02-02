struct Cat;

fn bar(_: &Cat) {
}

fn foo(cat: &mut Cat) {
    bar(&*cat);
}

pub fn main() {
    let mut mimi = ~Cat;
    foo(mimi);
}
