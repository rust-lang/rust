use dvec::DVec;

fn foo() -> int { 22 }

fn main() {
    let x = DVec::<@fn() -> int>();
    x.push(foo);
    assert (x[0])() == 22;
}