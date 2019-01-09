fn foo(_: &'static u32) {}

fn unpromotable<T>(t: T) -> T { t }

fn main() {
    foo(&unpromotable(5u32));
}
//~^^ ERROR borrowed value does not live long enough
