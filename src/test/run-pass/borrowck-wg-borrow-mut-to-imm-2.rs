struct Cat;

fn bar(_: &Cat) {
}

fn foo(cat: &mut Cat) {
    bar(&*cat);
}

fn main() {
    let mut mimi = ~Cat;
    foo(mimi);
}
