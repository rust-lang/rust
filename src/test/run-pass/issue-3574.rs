// xfail-test
// rustc --test match_borrowed_str.rs.rs && ./match_borrowed_str.rs
extern mod std;

fn compare(x: &str, y: &str) -> bool
{
    match x
    {
        "foo" => y == "foo",
        _ => y == "bar",
    }
}

#[test]
fn tester()
{
    assert compare("foo", "foo");
}
