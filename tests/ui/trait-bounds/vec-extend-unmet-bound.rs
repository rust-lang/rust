fn main() {
    let mut x = vec!["a", "b", "c"];
    x.extend([String::from("z")]);
    //~^ ERROR can't extend `Vec<&str>` with an iterator of `String` items
    //~| NOTE the trait `Extend<String>` is not implemented for `Vec<&str>`
    //~| NOTE you might have meant to extend a `Vec<String>` or pass in an `Iterator<Item = &str>`
    //~| HELP `Vec<T, A>` implements trait `Extend<A>`
    x.extend(String::from("z"));
    //~^ ERROR `String` is not an iterator
    //~| NOTE `String` is not an iterator; try calling `.chars()` or `.bytes()`
    //~| NOTE required by a bound introduced by this call
    //~| HELP the trait `Iterator` is not implemented for `String`
    //~| NOTE required for `String` to implement `IntoIterator`
    //~| NOTE required by a bound in `extend`
    x.extend(1);
    //~^ ERROR `{integer}` is not an iterator
    //~| NOTE `{integer}` is not an iterator
    //~| NOTE required by a bound introduced by this call
    //~| HELP the trait `Iterator` is not implemented for `{integer}`
    //~| NOTE required for `{integer}` to implement `IntoIterator`
    //~| NOTE required by a bound in `extend`
    //~| NOTE if you want to iterate between `start` until a value `end`
}
