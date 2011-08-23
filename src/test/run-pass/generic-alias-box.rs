

fn id<@T>(t: &T) -> T { ret t; }

fn main() {
    let expected = @100;
    let actual = id::<@int>(expected);
    log *actual;
    assert (*expected == *actual);
}
