struct S<'a, T:'a> {
    o: &'a Option<T>
}

fn main() {
    S { o: &None }; //~ ERROR type annotations needed [E0282]
}
