fn foo<T: Copy>(x: &T) -> T{
    match x {
        &a => a
    }
}

fn main() {
    assert foo(&3) == 3;
    assert foo(&'a') == 'a';
    assert foo(&@"Dogs rule, cats drool") == @"Dogs rule, cats drool";
}
